(define (problem pnp_on_table)
    (:domain blocks-tamp)
    (:objects
    block_r block_red1 - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_r workspace_table)
        (clear block_r)

        (on_table block_red1 workspace_table)
        (clear block_red1)

        (handempty robot)
        (gripper_at_init)

        (ori_of block_r o1)
        (ori_of block_red1 o1)
    )
    (:goal (and
        (on block_r block_red1)
    ))
)
