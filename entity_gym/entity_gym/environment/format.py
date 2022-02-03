VecObs(
    features={
        "Mine": RaggedBufferF32(
            [
                [[0, 2], [0, 1], [2, 2], [0, 0], [1, 0]],
                [[2.0, 1.0]],
                [[1, 0], [0, 1], [2, 2]],
            ]
        ),
        "Robot": RaggedBufferF32(
            [
                [[1.0, 1.0]],
                [[2.0, 0.0]],
                [[0, 0], [2, 0]],
            ]
        ),
        "Orbital Cannon": RaggedBuffer(
            [
                [],
                [[0.0]],
                [],
            ]
        ),
    },
    action_masks={
        "Move": VecCategoricalActionMask(
            actors=RaggedBufferI64(
                [
                    [[5]],
                    [[1]],
                    [[3], [4]],
                ]
            ),
            mask=RaggedBufferBool(
                [
                    [[True, True, True, True, True]],
                    [[False, True, True, False, True]],
                    [
                        [True, False, True, False, True],
                        [False, True, True, False, True],
                    ],
                ]
            ),
        ),
        "Fire Orbital Cannon": VecSelectEntityActionMask(
            actors=RaggedBufferI64(
                [
                    [],
                    [[2]],
                    [],
                ]
            ),
            actees=RaggedBufferI64(
                [
                    [],
                    [[0], [1]],
                    [],
                ]
            ),
        ),
    },
    reward=array([0.0, 0.0, 0.0], dtype=float32),
    done=array([False, False, False]),
    end_of_episode_info={},
)
