Observation(
    features={
        "Mine": array(
            [[1.0, 2.0], [0.0, 4.0], [2.0, 4.0], [1.0, 1.0], [5.0, 1.0]],
            dtype=float32,
        ),
        "Robot": array([[4.0, 0.0], [2.0, 2.0]], dtype=float32),
    },
    actions={
        "Move": CategoricalActionMask(
            actor_ids=None, actor_types=["Robot"], mask=None
        )
    },
    done=False,
    reward=0.0,
    ids={"Robot": [("Robot", 0), ("Robot", 1)]},
    end_of_episode_info=None,
)
Observation(
    features={
        "Mine": array(
            [[5.0, 0.0], [4.0, 0.0], [2.0, 1.0], [1.0, 0.0], [5.0, 1.0]],
            dtype=float32,
        ),
        "Robot": array([[0.0, 0.0], [4.0, 4.0]], dtype=float32),
    },
    actions={
        "Move": CategoricalActionMask(
            actor_ids=None, actor_types=["Robot"], mask=None
        )
    },
    done=False,
    reward=0.0,
    ids={"Robot": [("Robot", 0), ("Robot", 1)]},
    end_of_episode_info=None,
)