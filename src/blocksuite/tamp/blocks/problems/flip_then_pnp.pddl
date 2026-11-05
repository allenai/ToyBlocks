(define (problem flip-then-pnp)
    (:domain blocks-tamp)
    (:objects
        block_r - block_r_e
        block_red1 - block
        workspace_table - table
        robot - gripper
    )
    (:init
        (on_table block_r workspace_table)
        (on_table block_red1 workspace_table)
        (clear block_r)
        (clear block_red1)
        (handempty robot)
        (gripper_at_init)
        (ori_of block_r o17)
        (ori_of block_red1 o1)
    )
    (:goal (and
        (ori_of block_r o2)
        (on block_r block_red1)
    ))
    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
)
