(define (problem flip)
    (:domain blocks-tamp)
    (:objects
    block_r block_red1 block_blue - block
    workspace_table - table
    robot - gripper
    )
    (:init
        (on_table block_r workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_blue workspace_table)
        (clear block_r)
        (clear block_red1)
        (clear block_blue)
        (handempty robot)
        (gripper_at_init)

        (ori_of block_r o21)
        (ori_of block_red1 o21)
        (ori_of block_blue o21)
    )
    (:goal (and
        (ori_of block_r o2)
    ))
    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
    ;friction_range = [0.1, 0.4]
)