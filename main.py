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
            empty_steps = 0

            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents}
                s, r, done, info = env.step(actions)

                # cập nhật học
                for ts in s:
                    ql_agents[ts].learn(env.encode(s[ts], ts), r[ts])

                # kiểm tra xe đang hoạt động
                active_cars = len(env.sumo.vehicle.getIDList()) if hasattr(env, "sumo") else 1
                total_r = sum(r.values())

                if active_cars == 0 or total_r == 0:
                    empty_steps += 1
                else:
                    empty_steps = 0

                if empty_steps > 200:
                    print(f"Early stop at step {env.sim_step}: no cars for {empty_steps} steps")
                    env.save_csv(f"outputs/3x3/dqn_run{run}", episode)
                    break


    env.close()
