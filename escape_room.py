import numpy as np

import drawing


class EscapeRoom:
    def __init__(self, render=False, nb_agents=2, field_width=5, field_height=5, actions=None, fixed_attitude=0, random_wall=0, random_exit=0, random_lever=0, stay_reward_no_att=0, stay_reward_att=-1, lever_reward=-1, exit_reward=10):
        """
        Initialize the Escape Room Environment.
        :param render: Render escape room visually: "True" for yes, "False" for no
        :param nb_agents: number of agents
        :param field_width: width of the escape room
        :param field_height: height of the escape room
        :param actions: set of actions [X, Y]; X to move horizontal, Y to move vertical
        :param fixed_attitude: keep attitude fixed "True" or randomly "False"
        :param random_wall: generate wall randomly
        :param random_exit: generate exit randomly
        :param random_lever: generate lever randomly
        :param stay_reward_no_att: reward for staying for agent with no attitude
        :param stay_reward_att: reward for staying for agent with attitude
        :param lever_reward: reward for staying on lever
        :param exit_reward: reward for reaching exit
        """

        if actions is None:
            actions = [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]
        self.nb_agents = nb_agents
        self.field_width = field_width
        self.field_height = field_height
        self.actions = actions
        self.nb_actions = len(self.actions)
        self.render = render
        self.fixed_attitude = fixed_attitude
        self.random_wall = random_wall
        self.random_exit = random_exit
        self.random_lever = random_lever
        self.stay_reward_no_att = stay_reward_no_att
        self.stay_reward_att = stay_reward_att
        self.lever_reward = lever_reward
        self.exit_reward = exit_reward

        # render_objects: escape room visuals to render for humans
        self.render_objects = []

        # exit_open: door open: "True", door closed: "False"
        self.exit_open = False

        # agent with attitude left the room
        self.escape_room_done = False

        # agent who wants to leave the room has attitude "1", others "0"
        self.attitude = np.zeros(self.nb_agents, dtype=int)

        # escape room state
        self.world = np.zeros((self.field_width, self.field_height), dtype=int)
        self.wall = []
        self.exit_pos = np.zeros(2, dtype=int)
        self.lever_pos = np.zeros(2, dtype=int)
        self.agents_pos = np.zeros((self.nb_agents, 2), dtype=int)

        # escape room observation of all agents
        self.observations = []

        self.reset()

    def step(self, actions):
        """
        Executes actions of agents in the escape room.
        :param actions: list of actions the agents want to perform
        :return: self.observations: all agent observations, step_rewards: list of rewards for this step
        """

        # stores position to reset render objects
        old_positions = self.agents_pos.copy()
        step_rewards = np.zeros(self.nb_agents)
        queue = np.random.choice([0, self.nb_agents - 1], self.nb_agents, replace=False)

        # select random agent to perform action
        for agent_id in queue:
            if self.collision_check(agent_id, actions[agent_id]):
                self.agents_pos[agent_id] = [self.agents_pos[agent_id][0] + actions[agent_id][0], self.agents_pos[agent_id][1] + actions[agent_id][1]]

            # if agent with attitude reaches door, the episode ends
            if self.attitude[agent_id]:
                step_rewards[agent_id] += self.stay_reward_att
                if self.agents_pos[agent_id][0] == self.exit_pos[0] and self.agents_pos[agent_id][1] == self.exit_pos[1]:
                    step_rewards[agent_id] += self.exit_reward
                    self.escape_room_done = True

            # agents without attitude get punished for standing on lever but open the door
            else:
                if self.agents_pos[agent_id][0] == self.lever_pos[0] and self.agents_pos[agent_id][1] == self.lever_pos[1]:
                    step_rewards[agent_id] += self.lever_reward
                    self.world[self.exit_pos[0]][self.exit_pos[1]] = 0
                    if self.exit_pos in self.wall:
                        self.wall.remove([self.exit_pos[0], self.exit_pos[1]])
                    self.exit_open = True
                else:
                    step_rewards[agent_id] += self.stay_reward_no_att

        # update observations of agents
        self.observations = []
        for agent_id in range(self.nb_agents):
            self.observations.append(self.observation_one(agent_id))

        # update render objects
        if self.render:
            for i in range(len(self.render_objects)):
                self.render_objects[i].update_render(self, i, old_positions)

        return self.observations, step_rewards

    def reset(self):
        """
        Resets the escape room for new episodes.
        """

        # set default values
        self.exit_open = False
        self.escape_room_done = False
        self.wall = []
        self.world = np.zeros((self.field_width, self.field_height))
        self.exit_pos = np.full(2, -1, dtype=int)
        self.lever_pos = np.full(2, -1, dtype=int)

        # picks agent who has to exit fast
        self.attitude = np.zeros(self.nb_agents, dtype=int)
        if self.fixed_attitude:
            self.attitude[0] = 1
        else:
            self.attitude[np.random.randint(self.nb_agents)] = 1

        # generates position of wall clockwise
        if self.random_wall:
            wall_direction = np.random.randint(4)
        else:
            wall_direction = 2

        # real: left, visual: left
        if wall_direction is 0:
            for i in range(self.field_height):
                self.wall.append([0, i])
        # real: top, visual: bottom
        if wall_direction is 1:
            for i in range(self.field_width):
                self.wall.append([i, self.field_height - 1])
        # real: right, visual: right
        if wall_direction is 2:
            for i in range(self.field_height):
                self.wall.append([self.field_width - 1, i])
        # real: bottom, visual: top
        if wall_direction is 3:
            for i in range(self.field_width):
                self.wall.append([i, 0])

        for i in range(len(self.wall)):
            self.world[self.wall[i][0]][self.wall[i][1]] = 1

        # generates door locations
        if self.random_exit:
            self.exit_pos = self.wall[np.random.randint(len(self.wall))].copy()
        else:
            self.exit_pos = self.wall[int(len(self.wall)/2)].copy()

        # generates and ensures lever is not in front of the door
        if self.random_lever:
            next_door = [[self.exit_pos[0] + 1, self.exit_pos[1]], [self.exit_pos[0] - 1, self.exit_pos[1]],
                        [self.exit_pos[0], self.exit_pos[1] + 1], [self.exit_pos[0], self.exit_pos[1] - 1]]

            while -1 in self.lever_pos or self.lever_pos in next_door or self.lever_pos in self.wall:
                self.lever_pos = [np.random.randint(low=0, high=self.field_width), np.random.randint(low=0, high=self.field_height)]
        else:
            self.lever_pos = [int(self.field_width/2), int(self.field_height/2)]

        # generates unique agent positions
        self.agents_pos = np.full((self.nb_agents, 2), -1, dtype=int)
        # for i in range(len(self.agents_pos)):
        #     while -1 in self.agents_pos[i]:
        #         new_pos = [np.random.randint(low=0, high=self.field_width), np.random.randint(low=0, high=self.field_height)]
        #         if new_pos not in self.agents_pos and new_pos not in self.wall and not (new_pos[0] == self.lever_pos[0] and new_pos[1] == self.lever_pos[1]):
        #             self.agents_pos[i] = new_pos

        for i in range(len(self.agents_pos)):
            while -1 in self.agents_pos[i]:
                new_pos = [np.random.randint(low=1, high=self.field_width-1), np.random.randint(low=1, high=self.field_height-1)]
                if new_pos not in self.agents_pos and new_pos not in self.wall and not (new_pos[0] == self.lever_pos[0] and new_pos[1] == self.lever_pos[1]):
                    self.agents_pos[i] = new_pos

        # init observations
        self.observations = []
        for agent_id in range(self.nb_agents):
            self.observations.append(self.observation_one(agent_id))

        # init render objects
        if self.render:
            self.render_objects = []
            for agent_id in range(self.nb_agents):
                self.render_objects.append(drawing.EscRoomRender(env=self, agent_id=agent_id, display_size=480))

    def collision_check(self, agent_id, action):
        """
        Checks if agents new position would result in a undesirable state.
        :param agent_id: agent index
        :param action: action the agent wants to execute
        :return: "True" if no collision, "False" if collision
        """

        no_collision = True
        new_pos = [self.agents_pos[agent_id][0] + action[0], self.agents_pos[agent_id][1] + action[1]]
        if new_pos[0] >= self.field_width or new_pos[0] < 0:
            no_collision = False
        if new_pos[1] >= self.field_height or new_pos[1] < 0:
            no_collision = False
        if new_pos in self.wall:
            no_collision = False

        return no_collision

    def observation_one(self, agent_id):
        """
        Updates observation of agent_id.
        :param agent_id: agent index
        :return: new observation for agent with the index "agent_id" in form of np.array with one hot entries
        """

        # channels of observation
        world_obs = 0
        exit_obs = 1
        lever_obs = 2
        attitude_obs = 3
        self_obs = 4
        other_obs = 5

        channels = 6
        observation = np.zeros((channels, self.field_width, self.field_height))

        # escape room
        for i in range(len(self.wall)):
            observation[world_obs][self.wall[i][0]][self.wall[i][1]] = 1

        # door
        observation[exit_obs][self.exit_pos[0]][self.exit_pos[1]] = 1

        # lever
        observation[lever_obs][self.lever_pos[0]][self.lever_pos[1]] = 1

        # attitude
        observation[attitude_obs] += self.attitude[agent_id]

        # agent location with the index "agent_id"
        observation[self_obs][self.agents_pos[agent_id][0]][self.agents_pos[agent_id][1]] = 1

        # all other agent locations
        for i in range(self.nb_agents):
            if i != agent_id:
                observation[other_obs][self.agents_pos[i][0]][self.agents_pos[i][1]] = 1

        return observation


def init_escape_room(params):
    """
    Initializes escape room according to the parameters
    :param params: parameters of the run
    :return: escape room environment
    """

    actions = None

    # use custom action set from parameters
    if params.use_actions:
        actions = params.actions

    # creates escape room environment
    esc_room = EscapeRoom(False, params.nb_agents, params.field_width, params.field_height, actions,
                          params.fixed_attitude, params.random_wall, params.random_exit, params.random_lever,
                          params.stay_reward_no_att, params.stay_reward_att, params.lever_reward, params.exit_reward)

    return esc_room


if __name__ == '__main__':
    esc_room = EscapeRoom(False, 2, 5, 5, [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]], 0, 0, 0, 0)