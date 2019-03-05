require 'xlua'      ---lutm: This script is called in shell run_cpu, and run_gpu.sh.
require 'optim'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_episodes', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:option('-zmq_port', 5050, 'ZMQ port')
cmd:option('-mode', 'Shooter', 'Experiment domain')
cmd:option('-exp_folder', 'logs/', 'folder for logs')

cmd:text()

local opt = cmd:parse(arg)  -- lutm: opt stands for option.

if not dqn then             -- lutm: In the beginning, dqn is null.
    require "initenv"
end

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)       --- lutm: setup函数定义在initenv.lua中

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local n_finished_episodes
local eval_steps
local episode_reward

local state, reward, terminal = game_env:newGame()      ---lutm: terminal is a boolean variable.

print("Iteration ..", step)
local win = nil
while step < opt.steps do           -- lutm: 对步数循环
    xlua.progress(step, opt.steps)  -- lutm: 打印进度条？

    step = step + 1
    local action_index, query_index = agent:perceive(reward, state, terminal)

    -- game over? get next game!
    if not terminal then
        state, reward, terminal = game_env:step(game_actions[action_index], query_index)
    else    
        state, reward, terminal = game_env:newGame()        
    end
    
    if step % opt.prog_freq == 0 then       ---lutm: prog_freq stands for frequency of progress output.
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        agent:report()
        collectgarbage()
    end

    if step%1000 == 0 then collectgarbage() end     -- lutm: to finalize unused objects.

    -- evaluation
    if step % opt.eval_freq == 0 and step > learn_start then

        game_env:evalStart()
        state, reward, terminal = game_env:newGame()

        test_avg_Q = test_avg_Q or optim.Logger(paths.concat(opt.exp_folder , 'test_avgQ.log'))
        test_avg_R = test_avg_R or optim.Logger(paths.concat(opt.exp_folder , 'test_avgR.log'))        

        -- 整个测试集上的奖励
        total_reward = 0
        -- 整个测试集上的奖励数（小于等于step数）
        nrewards = 0
        -- 整个测试集上，已完成的情节数
        n_finished_episodes = 0
        -- each情节的累计奖励
        episode_reward = 0
        -- 已评估的步数
        eval_steps = 0

        local eval_time = sys.clock()
        print("Testing...")

        while true do   --- eval_steps=5000
            xlua.progress(n_finished_episodes, opt.eval_episodes)
            eval_steps = eval_steps + 1

            --- testing=true，不进行抽样、记忆等操作
            --- epsilong=0.0，不探索，直接greedy
            local action_index, query_index = agent:perceive(reward, state, terminal, true, 0.0)

            -- Play game in test mode 
            state, reward, terminal = game_env:step(game_actions[action_index], query_index)

            if eval_steps%1000 == 0 then collectgarbage() end

            -- record every reward
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                n_finished_episodes = n_finished_episodes + 1
                state, reward, terminal = game_env:newGame()

                if n_finished_episodes == opt.eval_episodes then
                    break
                end
            end
        end

        game_env:evalEnd()
        state, reward, terminal = game_env:newGame() --start new game

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        local ind = #reward_history+1
        -- 取平均数，转换为每个情节的累计奖励
        total_reward = total_reward/math.max(1, n_finished_episodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

        -- plotting graphs
        test_avg_R:add{['Average Reward'] = total_reward}
        test_avg_Q:add{['Average Q'] = agent.v_avg}
     
        test_avg_R:style{['Average Reward'] = '-'}; test_avg_R:plot()        
        test_avg_Q:style{['Average Q'] = '-'}; test_avg_Q:plot()

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = n_finished_episodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*eval_steps/eval_time,
            n_finished_episodes, nrewards))
    end

    if step % opt.save_freq == 0 or step == opt.steps then
        local s, a, o, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_o,
        agent.valid_r, agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end
        filename = filename
        torch.save(opt.exp_folder .. filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(opt.exp_folder .. filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_o, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, o, r, s2, term
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
        print('Saved:', opt.exp_folder .. filename .. '.t7')
        io.flush()
        collectgarbage()
    end
end
