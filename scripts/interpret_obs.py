import numpy as np

def get_goal_object(mission):
    """
    Interpret agent observations in terms relative object location.

    Currently only works for levels GoToObj, GoToLocal, GoToObjMaze and GoTo

    :param mission:
    :return:
    """

    COLOR_TO_IDX = {
        'red': 0,
        'green': 1,
        'blue': 2,
        'purple': 3,
        'yellow': 4,
        'grey': 5
    }

    OBJECT_TO_IDX = {
        'unseen': 0,
        'empty': 1,
        'wall': 2,
        'floor': 3,
        'door': 4,
        'locked_door': 5,
        'key': 6,
        'ball': 7,
        'box': 8,
        'goal': 9,
        'lava': 10
    }

    goal_object = mission.split()[-2:]
    encoded_goal = [OBJECT_TO_IDX[goal_object[1]], COLOR_TO_IDX[goal_object[0]]]
    return(encoded_goal)

def interpret_obs(obss):
    all_facts = []

    for obs in obss:
        obj_image = np.flip(obs['image'][:, :, 0], 0)
        color_image = np.flip(obs['image'][:, :, 1], 0)
        mission = obs['mission']
        comb_image = np.stack((obj_image, color_image), axis=2)
        goal = get_goal_object(mission)

        # object_ahead, object_left, object_right, object_invisible = False, False, False, False
        object_ahead, object_immediately_left, object_immediately_right, object_invisible = False, False, False, False

        agent_ahead = comb_image[3]
        agent_left = comb_image[4:]
        agent_immediately_left = comb_image[4:,6]
        agent_right = comb_image[:3]
        agent_immediately_right = comb_image[:3,6]

        if any((agent_ahead == goal).all(1)):
            object_ahead = True

        if any((agent_immediately_left == goal).all(1)):
            object_immediately_left = True

        if any((agent_immediately_right == goal).all(1)):
            object_immediately_right = True

        if not any((comb_image.reshape((49,2)) == goal).all(1)):
            object_invisible = True

        facts = {'object_ahead': object_ahead,
                 # 'object_left': object_left,
                 # 'object_right': object_right,
                 'object_immediately_left' : object_immediately_left,
                 'object_immediately_right' : object_immediately_right,
                 'object_invisible': object_invisible}

        all_facts += [facts]

    return(all_facts)

# o = np.zeros((7,7), dtype=int)
# o[:5,2:] = 1
# o[:5,1] = 2
# o[4,2:] = 2
# o[3,2] = 8
#
# print(o)
# f = interpret_obs(o)
# print(f)

# a = np.array([[1,2,3,4,6], [1,2,3,4,6]])
# print(a)
# if 6 in a:
#     print('6 in a')
#
# print(np.isin(a, [6,7,8]).flatten())
#
# if any(np.isin(a, [6,7,8]).flatten()):
#     print('any obj')


# if any(np.isin(agent_ahead, OBJECTS)):
#     object_ahead = True
#
# # if any(np.isin(agent_left, OBJECTS).flatten()):
# #     object_left = True
# #
# # if any(np.isin(agent_right, OBJECTS).flatten()):
# #     object_right = True
#
# if any(np.isin(agent_immediately_left, OBJECTS).flatten()):
#     object_immediately_left = True
#
# if any(np.isin(agent_immediately_right, OBJECTS).flatten()):
#     object_immediately_right = True
#
# if not any(np.isin(obs, OBJECTS).flatten()):
#     object_invisible = True