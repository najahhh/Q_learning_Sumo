from __future__ import absolute_import
from __future__ import print_function
from sumolib import checkBinary, net
import traci
import sys
import optparse
import numpy as np
import random
import time
from results import Plot


class Car:

    def __init__(self,network, nb_cars):
        self.reward = 1

        #Q-learning parameters
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

        #Environment variables
        self.spaceHeadway = {
            "min": 0.,
            "max": 150.,
            "decimals": 0,
            "nb_values": 6
        }
        """
        Dictionaries of values, values are in m/s.
        """
        self.relativeSpeed = {
            "min": -8.33,#-13.89,
            "max": 8.33,#13.89,
            "decimals": 2,
            "nb_values": 6
        }
        self.speed = {
            "min": 0.,
            "max": 13.89,
            "decimals": 2,
            "nb_values": 6
        }

        """
        Environment space
        """
        self.spaceHeadway_space = \
            dict(((round(i, self.spaceHeadway.get('decimals'))), iteration)
                 for iteration, i in
                 enumerate(np.linspace(self.spaceHeadway.get('min'),
                                       self.spaceHeadway.get('max'),
                                       self.spaceHeadway.get('nb_values'))))

        self.relativeSpeed_space = \
            dict(((round(i, self.relativeSpeed.get('decimals'))), iteration)
                 for iteration, i in
                 enumerate(np.linspace(self.relativeSpeed.get('min'),
                                       self.relativeSpeed.get('max'),
                                       self.relativeSpeed.get('nb_values'))))

        self.speed_space = \
            dict(((round(i, self.speed.get('decimals'))), iteration)
                 for iteration, i in
                 enumerate(np.linspace(self.speed.get('min'),
                                       self.speed.get('max'),
                                       self.speed.get('nb_values'))))

        self.action = [1, -1, 0]
        self.action_space = \
            dict((i, iteration)
                 for iteration, i in enumerate(self.action))

        ## Q-table initialized to zeros
        self.q = np.array(np.zeros([len(self.spaceHeadway_space),
                                    len(self.relativeSpeed_space),
                                    len(self.speed_space),
                                    len(self.action_space)]))

        self.network = net.readNet(network)

        self.nb_cars = nb_cars
        self.controlled_car_id = "controlled_car"

        self.generate_route_file()
        options = self.get_options()
        params = self.set_params(options)
        traci.start(params)

    """
    Helper functions
    """
    def update_reward_on_collision(self, reward_type):
        if reward_type == "collision":
            self.reward -= 10

        elif reward_type == "security_distance":
            self.reward = 0

    def update_reward_spaceHeadway(self, space_headway, speed, speed_limit):
        if speed_limit and speed > round(self.speed.get('max') + (1 / 3.6) * 5, 2):
            self.reward /= 10
        else:
            dist = np.absolute(np.min([round(space_headway, 0), self.spaceHeadway.get('max')])
                               - self.get_follower(self.controlled_car_id, self.spaceHeadway.get('max')))
            self.reward += (self.spaceHeadway.get('max') - dist) / self.spaceHeadway.get('max')
        self.reward = round(self.reward, 2)

    def update_speed(self, a, speed):
        return np.max([self.speed.get('min'), speed + a])

    def epsilon_decay(self, step):
        if step % 1000 == 0:
            self.epsilon = round(self.epsilon * 0.9, 6)

    def e_greedy_policy(self, d_t, ds_t, s_t):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)
        else:
            return np.argmax(self.q[
                self.spaceHeadway_space.get(d_t),
                self.relativeSpeed_space.get(ds_t),
                self.speed_space.get(s_t)])

    def valueToDiscrete(self, value, index):
        i_lower=None
        index_max=np.NINF
        index_min =np.inf

        for i in index:
            if i > index_max:
                index_max = i
            if i < index_min:
                index_min = i
            if value >= i:
                i_lower = i
            if index_min <= value < i:
                if (value - i_lower) >= (i - i_lower) / 2:
                    value = i
                else:
                    value = i_lower
                break

        return np.min([index_max, np.max([index_min, value])])

    """
    Q-learning algorithm 
    """

    def Q_learning(self, episodes):
        self.set_speed_mode(self.controlled_car_id, 0)
        state = None
        steps = 0
        reward_list=[]

        reward_type = "security_distance"
        speed_limit = True

        plt_data = {
            "collisions": [], "space_headway": [], "relative_speed": [], "speed": [], "steps": 0
        }

        while True:
            if state:

                plt_data["space_headway"].append(state.get("space_headway"))
                plt_data["relative_speed"].append(round(state.get("relative_speed") * 3.6, 0))
                plt_data["speed"].append(round(state.get("speed") * 3.6, 0))

                d_t, ds_t, s_t = \
                    self.valueToDiscrete(state.get('space_headway'), self.spaceHeadway_space), \
                    self.valueToDiscrete(state.get('relative_speed'), self.relativeSpeed_space), \
                    self.valueToDiscrete(state.get('speed'), self.speed_space)

                a = self.e_greedy_policy(d_t, ds_t, s_t)

                q_t = self.q[
                    self.spaceHeadway_space.get(d_t),
                    self.relativeSpeed_space.get(ds_t),
                    self.speed_space.get(s_t),
                    self.action_space.get(self.action[a])]

                update_speed = self.update_speed(self.action[a], state.get('speed'))
                self.set_speed(self.controlled_car_id, update_speed)
                self.simulation_step()
                next_state = self.get_state(self.controlled_car_id)

                q_max_t1 = None
                if self.is_collided(self.controlled_car_id):
                    self.update_reward_on_collision(reward_type)
                    self.set_speed(self.controlled_car_id, 0)
                    q_max_t1 = 0
                    state = None
                    plt_data["collisions"].append(steps)

                elif next_state:

                    if reward_type == "security_distance":
                        self.update_reward_spaceHeadway(next_state.get('space_headway'), next_state.get('speed'), speed_limit)

                    ###Track rewards
                    reward_list.append(self.reward)
                    if (steps + 1) % 200 == 0:
                         ave_reward = round(np.mean(reward_list), 4)
                         reward_list = []
                         print('Episode {} Average Reward: {}'.format(steps + 1, ave_reward))
                    
                    d_t1, ds_t1, s_t1 = \
                        self.valueToDiscrete(next_state.get('space_headway'), self.spaceHeadway_space), \
                        self.valueToDiscrete(next_state.get('relative_speed'), self.relativeSpeed_space), \
                        self.valueToDiscrete(next_state.get('speed'), self.speed_space)

                    q_max_t1 = np.max(self.q[
                                          self.spaceHeadway_space.get(d_t1),
                                          self.relativeSpeed_space.get(ds_t1),
                                          self.speed_space.get(s_t1)])

                    state = next_state

                if q_max_t1 is not None:
                    self.q[
                        self.spaceHeadway_space.get(d_t),
                        self.relativeSpeed_space.get(ds_t),
                        self.speed_space.get(s_t),
                        self.action_space.get(self.action[a])] = \
                        (1 - self.alpha) * q_t + self.alpha * (self.reward + self.gamma * q_max_t1)

                steps += 1
                self.epsilon_decay(steps)

            else:
                self.simulation_step()
                state = self.get_state(self.controlled_car_id)
                self.set_speed(self.controlled_car_id, 0)

            if steps > (episodes * 1000):
                time.sleep(.1)

            if steps == episodes * 1000:
                plt_data["steps"] = steps
                results = Plot(self, plt_data)
                results.plot_()


    #### ENVIRONMENT FUNCTIONS
    
    def set_speed_mode(self, car_id, mode):
        traci.vehicle.setSpeedMode(car_id, mode)

    def set_speed(self, car_id, speed):
        traci.vehicle.setSpeed(car_id, speed)

    def get_position(self, car_id):
        return traci.vehicle.getPosition(car_id)

    def get_distance(self, position_1, position_2):
        return np.sqrt(np.power(position_1[0] - position_2[0], 2)
                       + np.power(position_1[1] - position_2[1], 2))


    def get_leader_on_edge(self, car_id):
        return traci.vehicle.getLeader(car_id, dist=0.0)

    def get_follower(self, car_id, horizon):
        follower_dist = traci.vehicle.getFollower(car_id, dist=0.0)[1]
        return np.min([round(follower_dist, 0), horizon]) if follower_dist > 0 else horizon

    def get_edge_id(self, car_id):
        return traci.vehicle.getRoadID(car_id)

    def get_next_edge(self, edge_id):
        outgoing_edges = list(self.network.getEdge(edge_id).getOutgoing().keys())[0]
        return {
            "id": outgoing_edges.getID(),
            "from": outgoing_edges.getFromNode().getID(),
            "to": outgoing_edges.getToNode().getID()
        }

    def get_nb_cars_on_edge(self, edge_id):
        try:
            return traci.edge.getLastStepVehicleNumber(edge_id)
        except traci.exceptions.TraCIException as e:
            print(e)
            return None

    def get_edges_until_leader(self, car_id):
        e, next_edge, is_car_on_next_edge, edges = None, None, None, []
        e = self.get_edge_id(car_id)
        if e and not e[0] == ':':
            next_edge = self.get_next_edge(e)
            while not is_car_on_next_edge:
                edges.append(next_edge)
                is_car_on_next_edge = self.get_nb_cars_on_edge(next_edge.get("id"))
                next_edge = self.get_next_edge(next_edge.get("id"))
        return edges

    def get_cars_on_edge(self, edge_id):
        try:
            return traci.edge.getLastStepVehicleIDs(edge_id)
        except traci.exceptions.TraCIException as e:
            print(e)
            return None

    def get_leader_on_next_edge_id(self, edges):
        if edges:
            return self.get_cars_on_edge(edges[-1].get("id"))[0]
        else:
            return None

    def get_node_position(self, node_id):
        return traci.junction.getPosition(node_id)

    def get_leader_on_next_edge_distance(self, car_1_id, car_2_id, edges):
        if car_1_id and car_2_id and edges:
            distance = self.get_distance(self.get_position(car_1_id),
                                         self.get_node_position(edges[0].get("from")))
            for i in range(len(edges)):
                if i < len(edges) - 1:
                    distance += self.get_distance(self.get_node_position(edges[i].get("from")),
                                                  self.get_node_position(edges[i].get("to")))
                else:
                    distance += self.get_distance(self.get_node_position(edges[i].get("from")),
                                                  self.get_position(car_2_id))
            return distance
        else:
            return None

    def get_leader(self, car_id):
        leader = {'id': None, 'distance': None}
        leader_on_edge = self.get_leader_on_edge(car_id)
        if leader_on_edge:
            leader['id'] = leader_on_edge[0]
            leader['distance'] = leader_on_edge[1]
        else:
            edges = self.get_edges_until_leader(car_id)
            leader['id'] = self.get_leader_on_next_edge_id(edges)
            leader['distance'] = self.get_leader_on_next_edge_distance(car_id, leader.get('id'), edges)
        return leader

    def get_speed(self, car_id):
        return traci.vehicle.getSpeed(car_id)

    def get_relative_speed(self, car1_id, car2_id):
        return self.get_speed(car1_id) - self.get_speed(car2_id)

    def get_state(self, car_id):
        leader = self.get_leader(car_id)
        if leader.get('id') and leader.get('distance'):
            return {
                "speed": round(self.get_speed(car_id), 2),
                "relative_speed": round(self.get_relative_speed(car_id, leader.get('id')), 2),
                "space_headway": round(leader.get('distance'), 0)
            }
        else:
            return None

    def get_collisions(self):
        return traci.simulation.getCollidingVehiclesIDList()

    def is_collided(self, car_id):
        return car_id in self.get_collisions()

    def get_current_time(self):
        return traci.simulation.getCurrentTime()

    def reset(self):
        pass

    def simulation_step(self):
        traci.simulationStep()

    def simulation_close(self):
        traci.close()
        sys.stdout.flush()

    def generate_route_file(self):
        with open("network/road.rou.xml", "w") as routes:
            print('<routes>', file=routes)
            print(
                ' <vType accel="2.9" decel="7.5" id="sumo_car" type="passenger" length="4.3" minGap="30" maxSpeed="13.89" sigma="0.5" />',
                file=routes)
            print(
                ' <vType accel="2.9" decel="7.5" id="my_car" type="passenger" length="4.3" minGap="0" maxSpeed="27.89" departspeed="0" sigma="0.5" />',
                file=routes)
            print(' <route id="circle_route" edges="1to2 2to3"/>', file=routes)
            print(
                f' <flow id="carflow" type="sumo_car" beg="0" end="0" number="{str(self.nb_cars - 1)}" from="1to2" to="2to3"/>',
                file=routes)
            print(f' <vehicle depart="1" id="{self.controlled_car_id}" route="circle_route" type="my_car" color="1,0,0" />',
                  file=routes)
            print('</routes>', file=routes)

    def get_options(self):
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--nogui", action="store_true",
                              default=False, help="run the commandline version of sumo")

        opt_parser.add_option('--log', action="store_true",
                              default=False, help="verbose warning & error log to file")

        opt_parser.add_option('--debug', action="store_true",
                              default=False, help="run the debug mode")

        opt_parser.add_option('--remote', action="store_true",
                              default=False, help="run remote")

        opt, args = opt_parser.parse_args()
        return opt

    def set_params(self, options):
        if options.nogui:
            sumo_binary = checkBinary('sumo')
        else:
            sumo_binary = checkBinary('sumo-gui')

        params = [sumo_binary, "-c", "network/road.sumocfg"]

        if not options.nogui:
            params.append("-S")
            params.append("--gui-settings-file")
            params.append("network/gui-settings.cfg")
            # params.append("-v")

        if options.log:
            params.append("--message-log")
            params.append("log/message_log")

        if options.debug:
            params.append("--save-configuration")
            params.append("debug/debug.sumocfg")

        if options.remote:
            params.append("--remote-port")
            params.append("9999")
            traci.init(9999)

        return params

