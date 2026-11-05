(define (problem flip_a)
    (:domain blocks-tamp)
    (:objects
    block_a - block
    workspace_table - table
    robot - gripper
    )
    (:init
        (on_table block_a workspace_table)
        (clear block_a)
        (handempty robot)
        (gripper_at_init)

        (ori_of block_a o21)
    )
    (:goal (and
        (ori_of block_a o2)
    ))
    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
    ;friction_range = [0.1, 0.4]
)
