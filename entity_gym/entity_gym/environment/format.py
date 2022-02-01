Observation(
    features={
        "Mine": array([[0.0, 2.0]], dtype=float32),
        "Robot": array([[0.0, 1.0]], dtype=float32),
        "Orbital Cannon": array([[0.0]], dtype=float32),
    },
    actions={
        "Move": CategoricalActionMask(
            actor_ids=None,
            actor_types=["Robot"],
            mask=array([[True, False, True, True, True]]),
        ),
        "Fire Orbital Cannon": SelectEntityActionMask(
            actor_ids=None,
            actor_types=["Orbital Cannon"],
            actee_types=["Mine", "Robot"],
            actee_ids=None,
            mask=None,
        ),
    },
    done=False,
    reward=0.0,
    ids={
        "Mine": [("Mine", 0)],
        "Robot": [("Robot", 0)],
        "Orbital Cannon": [("Orbital Cannon", 0)],
    },
    end_of_episode_info=None,
)
Observation(
    features={
        "Mine": array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=float32),
        "Robot": array([[2.0, 0.0], [1.0, 2.0]], dtype=float32),
        "Orbital Cannon": array([], shape=(0, 1), dtype=float32),
    },
    actions={
        "Move": CategoricalActionMask(
            actor_ids=None,
            actor_types=["Robot"],
            mask=array(
                [[False, True, True, False, True], [True, True, False, True, True]]
            ),
        ),
        "Fire Orbital Cannon": SelectEntityActionMask(
            actor_ids=None,
            actor_types=[],
            actee_types=["Mine", "Robot"],
            actee_ids=None,
            mask=None,
        ),
    },
    done=False,
    reward=0.0,
    ids={
        "Mine": [("Mine", 0), ("Mine", 1), ("Mine", 2)],
        "Robot": [("Robot", 0), ("Robot", 1)],
        "Orbital Cannon": [],
    },
    end_of_episode_info=None,
)
