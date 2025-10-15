from agents.dqn import DQNAgent, DQNConfig, EpsilonGreedy
from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    runs = 30
    episodes = 4

    env = SumoEnvironment(
        net_file="scenarios/bk/vn.net.xml",
        route_file="scenarios/bk/full_routes.xml",
        additional_sumo_cmd="--additional-files=scenarios/bk/vtype.xml",
        use_gui=False,
        num_seconds=80000,
        min_green=5,
        delta_time=5,
        sumo_warnings=False
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()

        cfg = DQNConfig(gamma=0.99, lr=3e-4, batch_size=64, buffer_size=100000)
        expl = EpsilonGreedy(eps_start=1.0, eps_end=0.05, eps_decay=1e-4)

        ql_agents = {
            ts: DQNAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                config=cfg,
                exploration=expl,
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in env.ts_ids:
                    ql_agents[ts].set_state(env.encode(initial_states[ts], ts))

            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(
                        next_state=env.encode(s[agent_id], agent_id),
                        reward=r[agent_id],
                        done=done.get(agent_id, False),
                    )

            env.save_csv(f"outputs/4x4/dqn_run{run}", episode)

    env.close()
