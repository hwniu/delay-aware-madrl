import argparse
import torch
import numpy as np

from itertools import count

from environment.delay_environment import DelayEnvironment
from environment.delay_environment import Delay_State
from algorithm_agent.delay_aware_matd3 import Delay_Aware_Matd3
from algorithm_agent.reward_augment import reward_augment
from delay_data.lstm import load_lstm_model
from delay_data.lstm import delay_predict
from utils.logger import Logger
from algorithm_agent.state_augment import state_augment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delay MATD3 with Gate-XL Transformer (PyTorch)')
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--max_episodes', default=1000, type=int)
    parser.add_argument('--save_interval', default=50, type=int)
    parser.add_argument('--seed', '-s', type=int, default=4, help='Seed for Reproducibility purposes.')
    parser.add_argument('--gamma', default=0.99, type=float, help='the reward discount factor')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--episode_length', default=100, type=int)
    parser.add_argument('--memory_length', default=int(1e6), type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--episodes_before_train', default=100, type=int)
    parser.add_argument('--poisson_average', default=2, type=int, help='the OAI Delay Average')
    parser.add_argument('--tau', default=0.005, help="the soft update parameter ratio")
    parser.add_argument('--actor_target_update_interval', default=2, help="td3 algorithm delay update interval")
    parser.add_argument('--target_noise_scale', default=0.02, help="target actor noise for target critic")
    parser.add_argument('--explore_noise_scale', default=0.02, help="source actor noise for action explore")
    args = parser.parse_args()
    Logger.logger.debug("------------------------------------ The Delay-Aware MATD3 Algorithm System is started ------------------------------------")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ENV = DelayEnvironment()
    state_dim = ENV.env.get_environment_state_dim()
    action_dim = ENV.env.get_action_dim()

    D_MATD3 = Delay_Aware_Matd3(state_dim, action_dim, args)
    D_MATD3.load_model()
    last_action_delay_predict_model = load_lstm_model()
    episode = 0
    action_none = None
    total_step = 0
    while episode < args.max_episodes:
        episode += 1
        step = 0
        average_reward = 0
        # 传递延迟状态实例
        state = Delay_State(generation_time=episode * args.episode_length + step, state=ENV.env.reset(episode * args.episode_length + step), delay=0, reward=-27000)
        ENV.state_buffer.append(state)
        # 初始化初始动作时延
        if D_MATD3.get_last_three_action_float_delay() is not None:
            last_action_transmission_delay = delay_predict(data=D_MATD3.get_last_three_action_float_delay(),
                                                           lstm_model=last_action_delay_predict_model)
        else:
            last_action_transmission_delay = np.random.randint(1, 3)
        while True:
            if args.mode == "train":
                step += 1
                total_step += 1
                D_MATD3.set_time((episode - 1) * args.episode_length + step)
                ENV.set_time((episode - 1) * args.episode_length + step)
                # 生成延迟动作实例 state is instance
                aug_state = state_augment(state, D_MATD3.action_buffer, last_action_transmission_delay, D_MATD3.action_dim, D_MATD3.state_dim)
                last_action = D_MATD3.generate_action(aug_state)
                D_MATD3.add_sending_action_buffer(action=last_action)
                D_MATD3.action_buffer.append(last_action)
                execute_action_buffer = []
                for action_item in D_MATD3.sending_action_buffer:
                    if D_MATD3.now - action_item.generation_time >= action_item.delay:
                        execute_action_buffer.append(action_item)
                if len(execute_action_buffer) == 0:
                    last_state = ENV.improve_execute_action(action_none)
                else:
                    # last_state 需要考虑task_information_buffer
                    last_state = ENV.improve_execute_action(execute_action_buffer)
                    ENV.state_buffer.append(last_state)
                    ENV.add_sending_state_buffer(last_state)
                    D_MATD3.pop_sending_action_buffer(execute_action_buffer)

                # 接收state
                next_state_buffer = []
                next_state = None
                for state_item in ENV.sending_state_buffer:
                    if ENV.now - state_item.generation_time >= state_item.delay:
                        next_state_buffer.append(state_item)
                reward = reward_augment(next_state_buffer, args.gamma)
                average_reward = average_reward + reward
                if len(next_state_buffer) == 0:
                    next_state = state
                else:
                    # 更新智能体中action_buffer中known_delay字段
                    for state_item in next_state_buffer:
                        for action_index in range(len(D_MATD3.action_buffer)):
                            if D_MATD3.action_buffer[action_index].generation_time + D_MATD3.action_buffer[action_index].delay == state_item.generation_time and D_MATD3.action_buffer[action_index].known_delay == False:
                                D_MATD3.action_buffer[action_index].set_delay_know()
                    next_last_state_index = 0
                    for state_index in range(len(next_state_buffer)):
                        if next_state_buffer[state_index].generation_time > next_state_buffer[next_last_state_index].generation_time:
                            next_last_state_index = state_index
                    next_state = next_state_buffer[next_last_state_index]
                if D_MATD3.get_last_three_action_float_delay() is not None:
                    last_action_transmission_delay = delay_predict(data=D_MATD3.get_last_three_action_float_delay(),
                                                                   lstm_model=last_action_delay_predict_model)
                else:
                    last_action_transmission_delay = np.random.randint(1, 5)
                next_aug_state = state_augment(next_state, D_MATD3.action_buffer, last_action_transmission_delay, D_MATD3.action_dim, D_MATD3.state_dim)
                # 为了简化代码，在这里直接将生成动作的执行时间给置入缓冲区中
                D_MATD3.experience_buffer.push(ENV.now, state.state, state.generation_time, aug_state, last_action.action, next_aug_state, reward, None, last_action.generation_time + last_action.delay)
                state = next_state
                aug_state = next_aug_state
                Logger.logger.debug("D_MATD3: [Episode %05d] [step %d] reward %6.4f" % (episode, step, sum(reward) / reward.shape[0]))

                if args.episode_length < step:
                    critic_loss, actor_loss = D_MATD3.update_parameter(episode)
                    Logger.logger.debug("D_MATD3: [Episode %05d] reward %6.4f" % (episode, average_reward/args.episode_length))
                    if episode % args.save_interval == 0 and args.mode == "train":
                        D_MATD3.save_model(episode)
                    ENV.reset()
                    break
            elif args.mode == "test":
                step += 1
                total_step += 1
                D_MATD3.set_time((episode - 1) * args.episode_length + step)
                ENV.set_time((episode - 1) * args.episode_length + step)
                # 生成延迟动作实例 state is instance
                aug_state = state_augment(state, D_MATD3.action_buffer, last_action_transmission_delay, D_MATD3.action_dim, D_MATD3.state_dim)
                last_action = D_MATD3.generate_action(aug_state)
                D_MATD3.add_sending_action_buffer(action=last_action)
                D_MATD3.action_buffer.append(last_action)
                execute_action_buffer = []
                for action_item in D_MATD3.sending_action_buffer:
                    if D_MATD3.now - action_item.generation_time >= action_item.delay:
                        execute_action_buffer.append(action_item)
                if len(execute_action_buffer) == 0:
                    last_state = ENV.improve_execute_action(action_none)
                else:
                    # last_state 需要考虑task_information_buffer
                    last_state = ENV.improve_execute_action(execute_action_buffer)
                    ENV.state_buffer.append(last_state)
                    ENV.add_sending_state_buffer(last_state)
                    D_MATD3.pop_sending_action_buffer(execute_action_buffer)

                # 接收state
                next_state_buffer = []
                next_state = None
                for state_item in ENV.sending_state_buffer:
                    if ENV.now - state_item.generation_time >= state_item.delay:
                        next_state_buffer.append(state_item)
                reward = reward_augment(next_state_buffer, args.gamma)
                average_reward = average_reward + reward
                if len(next_state_buffer) == 0:
                    next_state = state
                else:
                    # 更新智能体中action_buffer中known_delay字段
                    for state_item in next_state_buffer:
                        for action_index in range(len(D_MATD3.action_buffer)):
                            if D_MATD3.action_buffer[action_index].generation_time + D_MATD3.action_buffer[action_index].delay == state_item.generation_time and D_MATD3.action_buffer[action_index].known_delay == False:
                                D_MATD3.action_buffer[action_index].set_delay_know()
                    next_last_state_index = 0
                    for state_index in range(len(next_state_buffer)):
                        if next_state_buffer[state_index].generation_time > next_state_buffer[next_last_state_index].generation_time:
                            next_last_state_index = state_index
                    next_state = next_state_buffer[next_last_state_index]
                if D_MATD3.get_last_three_action_float_delay() is not None:
                    last_action_transmission_delay = delay_predict(data=D_MATD3.get_last_three_action_float_delay(),
                                                                   lstm_model=last_action_delay_predict_model)
                else:
                    last_action_transmission_delay = np.random.randint(1, 5)
                next_aug_state = state_augment(next_state, D_MATD3.action_buffer, last_action_transmission_delay, D_MATD3.action_dim, D_MATD3.state_dim)
                # 为了简化代码，在这里直接将生成动作的执行时间给置入缓冲区中
                state = next_state
                aug_state = next_aug_state
                Logger.logger.debug("D_MATD3: [Episode %05d] [step %d] reward %6.4f" % (episode, step, sum(reward) / reward.shape[0]))