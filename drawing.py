import gizeh
import numpy as np


class EscRoomRender:
    def __init__(self, env, agent_id: int, display_size):
        """
        Initialize render objects in escape room fashion and as list.
        :param env: escape room environment
        :param agent_id: agent index
        :param display_size: display width for window render
        """

        # colors of environment objects
        self.colors = {
            'self_agent': (52/255, 151/255, 219/255, 1),
            'other_agents': (204/255, 204/255, 255/255, 1),
            'exit': (1, 1, 1, 0.4),
            'lever': (1, 1, 1, 0.4),
            'wall': (1, 1, 1, 0.05),
            'field': (1, 1, 1, 1)
        }

        if env.attitude[agent_id] == 0:
            self.colors['lever'] = (255/255, 102/255, 102/255, 1)

        # size of one agent view
        self.display_size = display_size

        # size of escape room tiles
        self.tile_size = int((display_size/len(env.observations[agent_id][0]))*0.75)

        # offset of escape room tiles
        self.offset = int(display_size/len(env.observations[agent_id][0]))

        # render objects of the escape room in the view of agent with index "agent_id"
        self.observation_squares = []
        self.observation_squares_list = []

        # resets complete escape room render
        for i in range(len(env.observations[agent_id][0])):
            new_row = []
            for j in range(len(env.observations[agent_id][i][0])):
                new_row.append(gizeh.square(l=self.tile_size, fill=self.colors['field'],
                                            xy=(i * self.tile_size + self.offset, j * self.tile_size + self.offset)))
            self.observation_squares.append(new_row)

        # wall render objects
        for i_wall in range(len(env.wall)):
            self.observation_squares[env.wall[i_wall][0]][env.wall[i_wall][1]] = gizeh.square(l=self.tile_size, fill=self.colors['wall'], xy=(env.wall[i_wall][0] * self.tile_size + self.offset, env.wall[i_wall][1] * self.tile_size + self.offset))

        # door render object
        self.observation_squares[env.exit_pos[0]][env.exit_pos[1]] = gizeh.square(
            l=self.tile_size, fill=self.colors['exit'], xy=(
                env.exit_pos[0] * self.tile_size + self.offset,
                env.exit_pos[1] * self.tile_size + self.offset))

        # lever render object
        self.observation_squares[env.lever_pos[0]][env.lever_pos[1]] = gizeh.square(
            l=self.tile_size, fill=self.colors['lever'], xy=(
                env.lever_pos[0] * self.tile_size + self.offset,
                env.lever_pos[1] * self.tile_size + self.offset))

        self.update_render(env, agent_id, env.agents_pos)

    def update_render(self, env, agent_id: int, old_positions: []):
        """
        Updates render objects according to the escape room environment for agent with index "agent_id".
        :param env: escape room environment
        :param agent_id: agent index
        :param old_positions: agents prior positions
        """

        self.observation_squares_list = []

        # reset agent render objects
        for i_agents in range(len(old_positions)):
            self.observation_squares[old_positions[i_agents][0]][old_positions[i_agents][1]] = gizeh.square(
                l=self.tile_size, fill=self.colors['field'], xy=(
                    old_positions[i_agents][0] * self.tile_size + self.offset,
                    old_positions[i_agents][1] * self.tile_size + self.offset))

        # door open render object
        if env.exit_open:
            self.observation_squares[env.exit_pos[0]][env.exit_pos[1]] = gizeh.square(
                l=self.tile_size, fill=self.colors['field'], xy=(
                    env.exit_pos[0] * self.tile_size + self.offset,
                    env.exit_pos[1] * self.tile_size + self.offset))
            self.colors['lever'] = (153/255, 255/255, 153/255, 1)

        # lever render object
        if env.lever_pos in old_positions:
            self.observation_squares[env.lever_pos[0]][env.lever_pos[1]] = gizeh.square(
                l=self.tile_size, fill=self.colors['lever'], xy=(
                    env.lever_pos[0] * self.tile_size + self.offset,
                    env.lever_pos[1] * self.tile_size + self.offset))

        # other agents render objects
        for i_agents in range(len(env.agents_pos)):
            if i_agents != agent_id:
                self.observation_squares[env.agents_pos[i_agents][0]][env.agents_pos[i_agents][1]] = gizeh.square(
                    l=self.tile_size, fill=self.colors['other_agents'], xy=(
                        env.agents_pos[i_agents][0] * self.tile_size + self.offset,
                        env.agents_pos[i_agents][1] * self.tile_size + self.offset))

        # self agent render object
        self.observation_squares[env.agents_pos[agent_id][0]][env.agents_pos[agent_id][1]] = gizeh.square(
            l=self.tile_size, fill=self.colors['self_agent'], xy=(
                env.agents_pos[agent_id][0] * self.tile_size + self.offset,
                env.agents_pos[agent_id][1] * self.tile_size + self.offset))

        # pack all objects in list in order to render
        for re_obj in self.observation_squares:
            self.observation_squares_list.extend(re_obj)


def trading_infos(info: [] = None, env=None):
    """
    Generates trading text and renders objects of the trading offers
    :param info: information about trading
    :param env: escape room environment
    :return: render_offers: offer render objects, render_texts: text information about trading
    """
    render_offers = []
    render_texts = []
    if info:
        offer_pos = np.zeros((env.nb_agents, 2), dtype=int)
        for i in range(env.nb_agents):
            info_text = []
            info_text.append(gizeh.text("agent {} offer: ({}, {})".format(i, info[0][i][0], info[0][i][1]),
                                        fontfamily="Arial", fontweight="bold", fill=(1, 1, 1), fontsize=15,
                                        xy=(60, 15), h_align='left'))
            if info[1][i]:
                info_text.append(
                    gizeh.text("agent {} pays agent {}: {:.3f}.".format(i, ((i + 1) % 2), info[2][i]),
                                            fontfamily="Arial", fontweight="bold", fill=(1, 1, 1), fontsize=15,
                                            xy=(60, 40), h_align='left'))
            else:
                info_text.append(
                    gizeh.text("no trade",
                               fontfamily="Arial", fontweight="bold", fill=(1, 1, 1), fontsize=15, xy=(60, 40),
                               h_align='left'))

            render_texts.append(info_text)

            if env.collision_check(i, info[0][((i + 1) % 2)]):
                offer_pos[i] = [env.agents_pos[i][0] + info[0][((i + 1) % 2)][0], env.agents_pos[i][1] + info[0][((i + 1) % 2)][1]]
            else:
                offer_pos[i] = [env.agents_pos[i][0], env.agents_pos[i][1]]

            render_offers.append(gizeh.square(l=env.render_objects[0].tile_size-5, fill=None, xy=(
                    offer_pos[i][0] * env.render_objects[0].tile_size + env.render_objects[0].offset,
                    offer_pos[i][1] * env.render_objects[0].tile_size + env.render_objects[0].offset), stroke=(0.2, 0.2, 0.2, 0.5), stroke_width=5))

    return render_offers, render_texts


def render_esc_room(combined_frames: [], render_objects: [], frames_per_state: int, info: [] = None, env=None):
    """
    Renders render objects of all agents and combined them vertically.
    :param combined_frames: frames until now
    :param render_objects: all render objects of all agents
    :param frames_per_state: amount of frames the state is rendered
    :param info: information about trading
    :param env: escape room environment
    :return: combined frames of current and past states
    """

    # print and show trading information
    render_offers, render_texts = trading_infos(info, env)

    # creates surface to draw on
    size = render_objects[0].display_size
    surfaces = [gizeh.Surface(size, size, (0.3, 0.3, 0.3)), gizeh.Surface(size, size, (0.3, 0.3, 0.3))]

    # list of all individual agent frames
    current_frame = []
    for agent_id in range(len(render_objects)):
        gizeh.Group(render_objects[agent_id].observation_squares_list).draw(surfaces[agent_id])

        if info:
            render_texts[agent_id][0].draw(surfaces[agent_id])
            render_texts[agent_id][1].draw(surfaces[agent_id])

            if info[0][(agent_id + 1) % 2][0] != 0.0 or info[0][(agent_id + 1) % 2][1] != 0.0:
                render_offers[agent_id].draw(surfaces[agent_id])

        current_frame.append(surfaces[agent_id].get_npimage())

    # combines all agent frames into one single frame
    # TODO combine frames for n agents rather than 2
    for i_frame in range(frames_per_state):
        combined_frames.append(np.append(current_frame[0], current_frame[1], axis=0))

    return combined_frames
