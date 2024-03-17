from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time

all_operators = ['park', 'move north', 'move south', 'move east', 'move west', 'pick up', 'charge', 'drop off']


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    MD = manhattan_distance
    robot = env.get_robot(robot_id)

    def package_points(package):
        return 2 * MD(package.position, package.destination)

    def cost_to_deliver(robot, package):
        return MD(robot.position, package.destination)

    def cost_to_pick(robot, package):
        return MD(robot.position, package.position)

    def min_dist_to_charger(robot, env):
        return min(MD(robot.position, charger.position) for charger in env.charge_stations)

    def battery_sufficient_for_packet_delivery(robot, package):
        return cost_to_deliver(robot, package) < robot.battery

    def any_reachable_packet(robot, env):
        return any(cost_to_pick(robot, package) < robot.battery for package in env.packages)

    def credit_diff(env, robot_id):
        robot = env.get_robot(robot_id)
        other_robot_id = 1 - robot_id
        other_robot = env.get_robot(other_robot_id)
        return robot.credit - other_robot.credit

    if env.done():
        return 10000 * credit_diff(env, robot_id)

    if robot.package and battery_sufficient_for_packet_delivery(robot, robot.package):
        return 1000 * credit_diff(env, robot_id) + 100 * package_points(robot.package) - cost_to_deliver(robot, robot.package)
    elif not robot.package and any_reachable_packet(robot, env):
        best_package = max(env.packages, key=package_points)
        return 1000 * credit_diff(env, robot_id) - cost_to_pick(robot, best_package)
    else:  # battery insufficient to deliver current packet or to pick a new packet
        battery_after_charge = robot.battery - min_dist_to_charger(robot, env) + robot.credit
        return 500 * credit_diff(env, robot_id) + battery_after_charge


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):

    def rb_minmax(self, env, agent_id, agent_turn_id, d, start_time, time_limit):
        if env.done() or d == 0 or time.time() - start_time > 0.95 * time_limit:
            return smart_heuristic(env, agent_id), None

        operators, children = self.successors(env, agent_turn_id)

        if agent_turn_id == agent_id:
            curr_max = float("-inf")
            step = random.choice(operators)
            for child, op in zip(children, operators):
                val, _ = self.rb_minmax(child, agent_id, 1 - agent_turn_id, d-1, start_time, time_limit)
                if val > curr_max:
                    step = op
                    curr_max = val

            return curr_max, step
        else:
            curr_min = float("inf")
            for child in children:
                val, _ = self.rb_minmax(child, agent_id, 1 - agent_turn_id, d-1, start_time, time_limit)
                curr_min = min(curr_min, val)

            return curr_min, None

    def anytime_minimax(self, env, agent_id, time_limit):
        start_time = time.time()
        operators, _ = self.successors(env, agent_id)
        chosen_operator = random.choice(operators)
        d = 1

        while time.time() - start_time < 0.95 * time_limit:
            _, op = self.rb_minmax(env, agent_id, agent_id, d, start_time, time_limit)
            if op is not None:
                chosen_operator = op
            d += 1

        return chosen_operator

    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        return self.anytime_minimax(env, agent_id, time_limit)


class AgentAlphaBeta(Agent):

    def rb_alpha_beta(self, env, agent_id, agent_turn_id, d, alpha, beta, start_time, time_limit):
        if env.done() or d == 0 or time.time() - start_time > 0.95 * time_limit:
            return smart_heuristic(env, agent_id), None

        operators, children = self.successors(env, agent_turn_id)

        if agent_turn_id == agent_id:
            curr_max = float("-inf")
            step = None
            for child, op in zip(children, operators):
                val, _ = self.rb_alpha_beta(child, agent_id, 1 - agent_turn_id, d-1, alpha, beta, start_time, time_limit)
                if val > curr_max:
                    step = op
                    curr_max = val
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return float("inf"), step

            return curr_max, step
        else:
            curr_min = float("inf")
            for child in children:
                val, _ = self.rb_alpha_beta(child, agent_id, 1 - agent_turn_id, d-1, alpha, beta, start_time, time_limit)
                curr_min = min(curr_min, val)
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    return float("-inf"), None

            return curr_min, None

    def anytime_rb_alpha_beta(self, env, agent_id, time_limit):
        start_time = time.time()
        operators, _ = self.successors(env, agent_id)
        chosen_operator = random.choice(operators)
        d = 1

        while time.time() - start_time < 0.95 * time_limit:
            _, op = self.rb_alpha_beta(env, agent_id, agent_id, d, float("-inf"), float("inf"), start_time, time_limit)
            if op is not None:
                chosen_operator = op
            d += 1

        return chosen_operator

    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        return self.anytime_rb_alpha_beta(env, agent_id, time_limit)


class AgentExpectimax(Agent):

    def cal_operators_prob(self, operators):
        num_of_high_prob_ops = int('pick up' in operators) + int('move east' in operators)
        num_of_uniform_prob_ops = len(operators) - num_of_high_prob_ops
        prob_of_uniform_op = 1 / (num_of_uniform_prob_ops + 2 * num_of_high_prob_ops)
        prob_of_high_op = 2 * prob_of_uniform_op

        op_to_prob = {op: prob_of_uniform_op for op in operators}
        if 'pick up' in operators:
            op_to_prob['pick up'] = prob_of_high_op
        if 'move east' in operators:
            op_to_prob['move east'] = prob_of_high_op

        return op_to_prob

    def rb_expectimax(self, env, agent_id, agent_turn_id, d, start_time, time_limit):
        if env.done() or d == 0 or time.time() - start_time > 0.95 * time_limit:
            return smart_heuristic(env, agent_id), None

        operators, children = self.successors(env, agent_turn_id)

        if agent_turn_id == agent_id:
            curr_max = float("-inf")
            step = None
            for child, op in zip(children, operators):
                val, _ = self.rb_expectimax(child, agent_id, 1 - agent_turn_id, d-1, start_time, time_limit)
                if val > curr_max:
                    step = op
                    curr_max = val

            return curr_max, step
        else:  # calculate weighted avg value of the opponent nodes values
            curr_val = 0
            op_to_prob = self.cal_operators_prob(operators)
            for child, op in zip(children, operators):
                val, _ = self.rb_expectimax(child, agent_id, 1 - agent_turn_id, d-1, start_time, time_limit)
                curr_val += op_to_prob[op] * val

            return curr_val, None

    def anytime_rb_expectimax(self, env, agent_id, time_limit):
        start_time = time.time()
        operators, _ = self.successors(env, agent_id)
        chosen_operator = random.choice(operators)
        d = 1

        while time.time() - start_time < 0.95 * time_limit:
            _, op = self.rb_expectimax(env, agent_id, agent_id, d, start_time, time_limit)
            if op is not None:
                chosen_operator = op
            d += 1

        return chosen_operator

    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        return self.anytime_rb_expectimax(env, agent_id, time_limit)


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)