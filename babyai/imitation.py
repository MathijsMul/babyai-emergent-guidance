import copy
import gym
import time
import datetime
import numpy as np
import sys
import itertools
import torch
from babyai.evaluate import batch_evaluate
import babyai.utils as utils
from babyai.rl import DictList
from babyai.model import ACModel
import multiprocessing
import os
import json
import logging

from sklearn.utils.class_weight import compute_class_weight


logger = logging.getLogger(__name__)

vocab_sizes_dict = utils.vocab_sizes_dict

class ImitationLearning(object):
    def __init__(self, args, ):
        self.args = args

        utils.seed(self.args.seed)

        # args.env is a list when training on multiple environments
        if getattr(args, 'multi_env', None):
            self.env = [gym.make(item) for item in args.multi_env]

            self.train_demos = []
            for demos, episodes in zip(args.multi_demos, args.multi_episodes):
                demos_path = utils.get_demos_path(demos, None, None, valid=False)
                logger.info('loading {} of {} demos'.format(episodes, demos))
                train_demos = utils.load_demos(demos_path)
                logger.info('loaded demos')
                if episodes > len(train_demos):
                    raise ValueError("there are only {} train demos in {}".format(len(train_demos), demos))
                self.train_demos.extend(train_demos[:episodes])
                logger.info('So far, {} demos loaded'.format(len(self.train_demos)))

            self.val_demos = []
            for demos, episodes in zip(args.multi_demos, [args.val_episodes] * len(args.multi_demos)):
                demos_path_valid = utils.get_demos_path(demos, None, None, valid=True)
                logger.info('loading {} of {} valid demos'.format(episodes, demos))
                valid_demos = utils.load_demos(demos_path_valid)
                logger.info('loaded demos')
                if episodes > len(valid_demos):
                    logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(valid_demos)))
                self.val_demos.extend(valid_demos[:episodes])
                logger.info('So far, {} valid demos loaded'.format(len(self.val_demos)))

            logger.info('Loaded all demos')

            observation_space = self.env[0].observation_space
            action_space = self.env[0].action_space

            vocab_max_size = sum([vocab_sizes_dict[env] for env in args.multi_env])

        else:
            self.env = gym.make(self.args.env)

            demos_path = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=False)
            demos_path_valid = utils.get_demos_path(args.demos, args.env, args.demos_origin, valid=True)

            logger.info('loading demos')
            self.train_demos = utils.load_demos(demos_path)
            logger.info('loaded demos')
            if args.episodes:
                if args.episodes > len(self.train_demos):
                    raise ValueError("there are only {} train demos".format(len(self.train_demos)))
                self.train_demos = self.train_demos[:args.episodes]

            self.val_demos = utils.load_demos(demos_path_valid)
            if args.val_episodes > len(self.val_demos):
                logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(self.val_demos)))
            self.val_demos = self.val_demos[:self.args.val_episodes]

            observation_space = self.env.observation_space
            action_space = self.env.action_space

            # Set max vocab size to number of distinct words occurring in level
            vocab_max_size = vocab_sizes_dict[args.env]

        if self.args.pretrained_corrector:
            print('Loading vocab from pretrained corrector')
            loaded_vocab = utils.format.Vocabulary(self.args.pretrained_corrector)
            loaded_vocab.path = os.path.join(utils.get_model_dir(self.args.model), "vocab.json")
            vocab_max_size += len(loaded_vocab) # make sure embedding matrix is big enough

        # vocabulary also loaded here
        self.obss_preprocessor = utils.ObssPreprocessor(args.model, observation_space,
                                                        getattr(self.args, 'pretrained_model', None),
                                                        vocab_max_size)

        if self.args.pretrained_corrector:
            self.obss_preprocessor.vocab = loaded_vocab
            self.obss_preprocessor.instr_preproc.vocab = loaded_vocab

        # Define actor-critic model
        self.model = utils.load_model(args.model, raise_not_found=False)
        if self.model is None:
            if getattr(self.args, 'pretrained_model', None):
                self.model = utils.load_model(args.pretrained_model, raise_not_found=True)

                if self.args.learner:
                    # in case a higher level requires more word embeddings, just add 15 extra rows to matrix (should be enough for sure)
                    extra_emb = torch.zeros((15, self.model.instr_dim), requires_grad=True)
                    extra_emb.data.normal_(0, 1)
                    extra_emb.data *= 1 / torch.sqrt(extra_emb.data.pow(2).sum(1, keepdim=True))
                    new_emb = torch.nn.Parameter(torch.cat((self.model.word_embedding.weight.data, extra_emb)))
                    self.model.word_embedding.weight = new_emb
                    self.model.word_embedding.num_embeddings += 15

                self.obss_preprocessor.instr_preproc.vocab.max_size += 15

                self.model.vocab = self.obss_preprocessor.vocab

                self.model.use_learner = self.args.learner
                self.model.use_corrector = self.args.corrector
                self.model.use_critic = False

                if self.args.pretrained_corrector:
                    self.model.load_pretrained_corrector(self.args.pretrained_corrector)

            else:
                self.model = ACModel(obs_space=self.obss_preprocessor.obs_space,
                                            action_space=action_space,
                                            image_dim=args.image_dim,
                                            memory_dim=args.memory_dim,
                                            instr_dim=args.instr_dim,
                                            use_instr=not self.args.no_instr,
                                            lang_model=self.args.instr_arch,
                                            use_memory=not self.args.no_mem,
                                            arch=self.args.arch,
                                            vocabulary=self.obss_preprocessor.vocab,
                                            learner=self.args.learner,
                                            corrector=self.args.corrector,
                                            corr_length=self.args.corr_length,
                                            corr_own_vocab=self.args.corr_own_vocab,
                                            corr_vocab_size=self.args.corr_vocab_size,
                                            pretrained_corrector=self.args.pretrained_corrector,
                                            dropout=self.args.dropout,
                                            corrector_frozen=self.args.corrector_frozen,
                                            random_corrector=self.args.random_corrector,
                                            var_corr_len=self.args.var_corr_length,
                                            weigh_corrections=self.args.weigh_corrections)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr, eps=self.args.optim_eps)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        utils.save_model(self.model, args.model)
        self.obss_preprocessor.vocab.save()

        self.model.train()
        if torch.cuda.is_available():
            self.model.cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.args.class_weights:
            # compute class weights
            actions = np.array([action for demo in self.train_demos for action in demo[-1]])
            class_weights = compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(actions),
                                                      y=actions)
            self.class_weights = np.clip(class_weights, 0.0, 3.0)

        num_classes = 7
        self.confusion_matrix = torch.zeros(num_classes, num_classes)

    @staticmethod
    def default_model_name(args):
        if getattr(args, 'multi_env', None):
            # It's better to specify one's own model name for this scenario
            named_envs = '-'.join(args.multi_env)
        else:
            named_envs = args.env

        # Define model name
        suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        instr = args.instr_arch if args.instr_arch else "noinstr"
        model_name_parts = {
            'envs': named_envs,
            'pretrained': '' if args.pretrained_corrector is None else 'pretrained',
            'corrector': 'corr' if args.corrector else 'no_corr',
            'vocab' : 'own-vocab' if args.corr_own_vocab else 'instr-vocab',
            'arch': args.arch,
            'instr': instr,
            'seed': args.seed,
            'suffix': suffix}
        default_model_name = "{envs}_IL_{pretrained}{corrector}_{vocab}_{arch}_{instr}_seed{seed}_{suffix}".format(**model_name_parts)
        if getattr(args, 'pretrained_model', None):
            default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
        return default_model_name

    def starting_indexes(self, num_frames):
        if num_frames % self.args.recurrence == 0:
            return np.arange(0, num_frames, self.args.recurrence)
        else:
            if num_frames >= self.args.recurrence:
                return np.arange(0, num_frames, self.args.recurrence)[:-1]
            else:
                raise ValueError('Nr of frames is smaller than recurrence.')

    def run_epoch_recurrence(self, demos, is_training=False, show_error_dist=False, compute_cic=False):
        indices = list(range(len(demos)))
        if is_training:
            np.random.seed()
            np.random.shuffle(indices)
        batch_size = min(self.args.batch_size, len(demos))
        offset = 0

        if not is_training:
            self.model.eval()

        # Log dictionary
        log = {"entropy": [], "policy_loss": [], "accuracy": [], "correction_weight_loss": []}

        if compute_cic:
            log["cic"] = torch.tensor([], device=self.device)

        start_time = time.time()
        frames = 0
        for batch_index in range(len(indices) // batch_size):
            logger.info("batch {}, FPS so far {: .3f}".format(
                batch_index, frames / (time.time() - start_time) if frames else 0))
            batch = [demos[i] for i in indices[offset: offset + batch_size]]
            frames += sum([len(demo[3]) for demo in batch])

            _log = self.run_epoch_recurrence_one_batch(batch, is_training=is_training, show_error_dist=show_error_dist, compute_cic=compute_cic)

            log["entropy"].append(_log["entropy"])
            log["policy_loss"].append(_log["policy_loss"])
            log["accuracy"].append(_log["accuracy"])

            log_info = [_log["entropy"], _log["policy_loss"], _log["accuracy"]]

            if compute_cic:
                log["cic"] = torch.cat((log["cic"], _log["cic"]))

            if self.args.weigh_corrections:
                log["correction_weight_loss"].append(_log["correction_weight_loss"])
                log_info += [_log["correction_weight_loss"]]
                logger.info("H {:.3f} | pL {: .3f} | A {: .3f} | CW {: .3f}".format(*log_info))
            else:
                logger.info("H {:.3f} | pL {: .3f} | A {: .3f}".format(*log_info))

            offset += batch_size

        if not is_training:
            self.model.train()

        if show_error_dist:
            print(self.confusion_matrix)
            print(self.confusion_matrix.diag() / self.confusion_matrix.sum(1))

        if compute_cic:
            log["cic"] = torch.mean(log["cic"]).item()

        return log

    def run_epoch_recurrence_one_batch(self, batch, is_training=False, show_error_dist=False, compute_cic=False):
        batch = utils.demos.transform_demos(batch)
        batch.sort(key=len, reverse=True)
        # Constructing flat batch and indices pointing to start of each demonstration
        flat_batch = []
        inds = [0]

        for demo in batch:
            flat_batch += demo
            inds.append(inds[-1] + len(demo))

        flat_batch = np.array(flat_batch)
        inds = inds[:-1]

        num_frames = len(flat_batch)

        mask = np.ones([len(flat_batch)], dtype=np.float64)
        mask[inds] = 0
        mask = torch.tensor(mask, device=self.device, dtype=torch.float).unsqueeze(1)

        # Observations, true action, values and done for each of the stored demostration
        obss, action_true, done = flat_batch[:, 0], flat_batch[:, 1], flat_batch[:, 2]
        action_true = torch.tensor([action for action in action_true], device=self.device, dtype=torch.long)

        # Memory to be stored
        memories = torch.zeros([len(flat_batch), self.model.memory_size], device=self.device)
        episode_ids = np.zeros(len(flat_batch))
        memory = torch.zeros([len(batch), self.model.memory_size], device=self.device)

        if self.args.corrector:
            memories_corr = torch.zeros([len(flat_batch), self.model.memory_size], device=self.device)
            memory_corrector = torch.zeros([len(batch), self.model.memory_size], device=self.device)

        preprocessed_first_obs = self.obss_preprocessor(obss[inds], device=self.device)

        if self.args.learner:
            instr_embedding = self.model._get_instr_embedding(preprocessed_first_obs.instr)

        if self.args.weigh_corrections:
            # what is the best way to initialize the entropies?
            entropies = torch.zeros([len(flat_batch), 1], device=self.device)
            entropies_step = torch.zeros([len(batch), 1], device=self.device)

        cic_values = torch.tensor([], device=self.device)

        # Loop terminates when every observation in the flat_batch has been handled
        while True:
            # taking observations and done located at inds
            obs = obss[inds]
            done_step = done[inds]

            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)

            with torch.no_grad():
                # taking the memory till len(inds), as demos beyond that have already finished

                if compute_cic:
                    cic = self.model.compute_cic(obs=preprocessed_obs,
                                                memory=memory[:len(inds), :] if self.args.learner else None,
                                                instr_embedding=instr_embedding[:len(inds)] if self.args.learner else None,
                                                memory_corrector=memory_corrector[:len(inds), :] if self.args.corrector else None,
                                                previous_entropy=entropies_step[:len(inds), :] if self.args.weigh_corrections else None)
                    cic_values = torch.cat((cic_values, cic))

                model_forward = self.model(obs=preprocessed_obs,
                                            memory=memory[:len(inds), :] if self.args.learner else None,
                                            instr_embedding=instr_embedding[:len(inds)] if self.args.learner else None,
                                            memory_corrector=memory_corrector[:len(inds), :] if self.args.corrector else None,
                                            previous_entropy=entropies_step[:len(inds), :] if self.args.weigh_corrections else None)

                new_memory, new_memory_corrector, new_entropy = model_forward['memory'], model_forward['memory_corrector'], model_forward['dist'].entropy().unsqueeze(1)

                if self.args.show_corrections:
                    def process_correction(correction):
                        processed = []
                        len_corr = len(correction)
                        for idx, token in enumerate(correction):
                            if token == '<eos>':
                                processed += ['..' for i in range(len_corr - idx)]
                                break
                            else:
                                processed += [token]
                        return(processed)

                    correction_messages = model_forward['corrections'][:len(inds)]

                    if self.args.var_corr_length:
                        correction_messages = [process_correction(correction) for correction in correction_messages]
                    print(correction_messages)

            if self.args.learner:
                memories[inds, :] = memory[:len(inds), :]
                memory[:len(inds), :] = new_memory
            if self.args.corrector:
                memories_corr[inds, :] = memory_corrector[:len(inds), :]
                memory_corrector[:len(inds), :] = new_memory_corrector
            if self.args.weigh_corrections:
                entropies[inds, :] = entropies_step[:len(inds), :]
                entropies_step[:len(inds), :] = new_entropy

            episode_ids[inds] = range(len(inds))

            # Updating inds, by removing those indices corresponding to which the demonstrations have finished
            inds = inds[:len(inds) - sum(done_step)]
            if len(inds) == 0:
                break

            # Incrementing the remaining indices
            inds = [index + 1 for index in inds]

        # Here, actual backprop upto args.recurrence happens
        final_loss = 0
        final_entropy, final_policy_loss, final_value_loss = 0, 0, 0
        if self.args.weigh_corrections:
            final_correction_weight_loss = 0

        indexes = self.starting_indexes(num_frames)

        memory = memories[indexes]
        if self.args.corrector:
            memory_corrector = memories_corr[indexes]
        if self.args.weigh_corrections:
            prev_entropies = entropies[indexes]

        accuracy = 0
        total_frames = len(indexes) * self.args.recurrence

        for _ in range(self.args.recurrence):
            obs = obss[indexes]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            action_step = action_true[indexes]
            mask_step = mask[indexes]

            model_results = self.model(
                obs=preprocessed_obs,
                memory=memory * mask_step if self.args.learner else None,
                instr_embedding=instr_embedding[episode_ids[indexes]] if self.args.learner else None,
                memory_corrector=memory_corrector * mask_step if self.args.corrector else None,
                previous_entropy=prev_entropies if self.args.weigh_corrections else None)

            dist = model_results['dist']

            memory = model_results['memory']
            if self.args.corrector:
                memory_corrector = model_results['memory_corrector']
            if self.args.weigh_corrections:
                prev_entropies = dist.entropy().unsqueeze(1)

            entropy = dist.entropy().mean()

            if self.args.class_weights:
                nll = -dist.log_prob(action_step)
                device = torch.device("cuda" if action_step.is_cuda else "cpu")
                weights = torch.tensor(self.class_weights[action_step.cpu()], device=device).float()
                policy_loss = (nll * weights).mean()
            else:
                policy_loss = -dist.log_prob(action_step).mean()

            loss = policy_loss - self.args.entropy_coef * entropy

            if self.args.var_corr_length:
                loss += self.args.corr_loss_coef * model_results['correction_loss']

            if self.args.weigh_corrections:
                # penalize high corrector weight
                correction_weight_loss = model_results['correction_weight'].mean()
                loss += self.args.correction_weight_loss_coef * correction_weight_loss

            # additional penalty term for low entropy corrections
            # loss = policy_loss - self.args.entropy_coef * (entropy + correction_entropy)

            action_pred = dist.probs.max(1, keepdim=True)[1]
            accuracy += float((action_pred == action_step.unsqueeze(1)).sum()) / total_frames

            if show_error_dist:
                # fill confusion matrix
                # show accuracy per class to spot when the model is just applying a majority vote
                for true, pred in zip(action_step.view(-1), action_pred.view(-1)):
                    self.confusion_matrix[true.long(), pred.long()] += 1

            final_loss += loss
            final_entropy += entropy
            final_policy_loss += policy_loss
            if self.args.weigh_corrections:
                final_correction_weight_loss += correction_weight_loss
            indexes += 1

        final_loss /= self.args.recurrence

        if is_training:
            self.optimizer.zero_grad()
            final_loss.backward()

            self.optimizer.step()

        log = {}
        log["entropy"] = float(final_entropy / self.args.recurrence)
        log["policy_loss"] = float(final_policy_loss / self.args.recurrence)
        if self.args.weigh_corrections:
            log["correction_weight_loss"] = float(final_correction_weight_loss / self.args.recurrence)
        log["accuracy"] = float(accuracy)

        if compute_cic:
            log["cic"] = cic_values

        return log

    def validate(self, episodes, verbose=True):
        # Seed needs to be reset for each validation, to ensure consistency
        utils.seed(self.args.val_seed)

        if verbose:
            logger.info("Validating the model")
        if getattr(self.args, 'multi_env', None):
            agent = utils.load_agent(self.env[0], model_name=self.args.model, argmax=True)
        else:
            agent = utils.load_agent(self.env, model_name=self.args.model, argmax=True)

        # Setting the agent model to the current model
        agent.model = self.model

        agent.model.eval()
        logs = []

        for env_name in ([self.args.env] if not getattr(self.args, 'multi_env', None)
                         else self.args.multi_env):
            logs += [batch_evaluate(agent, env_name, self.args.val_seed, episodes)]
        agent.model.train()

        return logs

    def collect_returns(self):
        logs = self.validate(episodes=self.args.eval_episodes, verbose=False)
        mean_return = {tid: np.mean(log["return_per_episode"]) for tid, log in enumerate(logs)}
        return mean_return

    def train(self, train_demos, writer, csv_writer, status_path, header, reset_status=False):
        # Load the status
        def initial_status():
            return {'i': 0,
                    'num_frames': 0,
                    'patience': 0}

        status = initial_status()
        if os.path.exists(status_path) and not reset_status:
            with open(status_path, 'r') as src:
                status = json.load(src)
        elif not os.path.exists(os.path.dirname(status_path)):
            # Ensure that the status directory exists
            os.makedirs(os.path.dirname(status_path))

        # If the batch size is larger than the number of demos, we need to lower the batch size
        if self.args.batch_size > len(train_demos):
            self.args.batch_size = len(train_demos)
            logger.info("Batch size too high. Setting it to the number of train demos ({})".format(len(train_demos)))

        # Model saved initially to avoid "Model not found Exception" during first validation step
        utils.save_model(self.model, self.args.model)

        # best mean return to keep track of performance on validation set
        best_success_rate, patience, i = 0, 0, 0
        total_start_time = time.time()

        while status['i'] < getattr(self.args, 'epochs', int(1e9)):
            if 'patience' not in status:  # if for some reason you're finetuining with IL an RL pretrained agent
                status['patience'] = 0
            # Do not learn if using a pre-trained model that already lost patience
            if status['patience'] > self.args.patience:
                break
            if status['num_frames'] > self.args.frames:
                break

            status['i'] += 1
            i = status['i']
            update_start_time = time.time()

            # Learning rate scheduler
            self.scheduler.step()

            log = self.run_epoch_recurrence(train_demos, is_training=True)
            total_len = sum([len(item[3]) for item in train_demos])
            status['num_frames'] += total_len

            update_end_time = time.time()

            # save vocab
            self.obss_preprocessor.vocab.save() #path=self.args.model)

            # Print logs
            if status['i'] % self.args.log_interval == 0:
                total_ellapsed_time = int(time.time() - total_start_time)

                fps = total_len / (update_end_time - update_start_time)
                duration = datetime.timedelta(seconds=total_ellapsed_time)

                for key in log:
                    log[key] = np.mean(log[key])

                train_data = [status['i'], status['num_frames'], fps, total_ellapsed_time,
                              log["entropy"], log["policy_loss"], log["accuracy"]]

                if self.args.weigh_corrections:
                    logger.info(
                        "U {} | F {:06} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | A {: .3f} | CW {: .3f} ".format(*train_data + [log["correction_weight_loss"]]))
                else:
                    logger.info(
                        "U {} | F {:06} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | A {: .3f}".format(*train_data))

                # Log the gathered data only when we don't evaluate the validation metrics. It will be logged anyways
                # afterwards when status['i'] % self.args.val_interval == 0
                if status['i'] % self.args.val_interval != 0:
                    # instantiate a validation_log with empty strings when no validation is done
                    validation_data = [''] * len([key for key in header if 'valid' in key])

                    extra_info = []
                    if self.args.weigh_corrections:
                        extra_info += [log["correction_weight_loss"]]

                    assert len(header) == len(train_data + validation_data + extra_info)
                    if self.args.tb:
                        for key, value in zip(header, train_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    csv_writer.writerow(train_data + validation_data + extra_info)

            if status['i'] % self.args.val_interval == 0:
                # TODO: Check this, seems weird: valid_log generates own episodes; val_log runs on pre-generated val demos.
                #   Shouldn't both apply to the same demos?

                valid_log = self.validate(self.args.val_episodes)

                mean_return = [np.mean(log['return_per_episode']) for log in valid_log]
                success_rate = [np.mean([1 if r > 0 else 0 for r in log['return_per_episode']]) for log in
                                valid_log]

                val_log = self.run_epoch_recurrence(self.val_demos, show_error_dist=False, compute_cic=self.args.compute_cic)
                validation_accuracy = np.mean(val_log["accuracy"])

                if status['i'] % self.args.log_interval == 0:
                    validation_data = [validation_accuracy] + mean_return + success_rate
                    logger.info(("Validation: A {: .3f} " + ("| R {: .3f} " * len(mean_return) +
                                                             "| S {: .3f} " * len(success_rate))
                                 ).format(*validation_data))

                    extra_info = []
                    if self.args.weigh_corrections:
                        extra_info += [log["correction_weight_loss"]]

                    if self.args.compute_cic:
                        extra_info += [val_log["cic"]]

                    assert len(header) == len(train_data + validation_data + extra_info)
                    if self.args.tb:
                        for key, value in zip(header, train_data + validation_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    csv_writer.writerow(train_data + validation_data + extra_info)

                # In case of a multi-env, the update condition would be "better mean success rate" !
                if np.mean(success_rate) > best_success_rate:
                    best_success_rate = np.mean(success_rate)
                    status['patience'] = 0
                    with open(status_path, 'w') as dst:
                        json.dump(status, dst)
                    # Saving the model
                    logger.info("Saving best model")

                    if torch.cuda.is_available():
                        self.model.cpu()
                    utils.save_model(self.model, self.args.model + "_best")
                    self.obss_preprocessor.vocab.save(utils.get_vocab_path(self.args.model + "_best"))
                    if torch.cuda.is_available():
                        self.model.cuda()
                else:
                    status['patience'] += 1
                    logger.info(
                        "Losing patience, new value={}, limit={}".format(status['patience'], self.args.patience))

                if torch.cuda.is_available():
                    self.model.cpu()

                if self.args.save_each_epoch:
                    utils.save_model(self.model, self.args.model + '_epoch' + str(status['i']))
                else:
                    utils.save_model(self.model, self.args.model)

                if torch.cuda.is_available():
                    self.model.cuda()
                with open(status_path, 'w') as dst:
                    json.dump(status, dst)