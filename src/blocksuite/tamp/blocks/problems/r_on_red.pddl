(define (problem letter-color-10-20260629_120032_156554_8935)
    (:domain blocks-tamp)
    (:objects
    block_r - block_r_e
    block_red1 block_blue block_blue2 block_blue3 - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_r workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_blue workspace_table)
        (on_table block_blue2 workspace_table)
        (on_table block_blue3 workspace_table)
        (clear block_r)
        (clear block_red1)
        (clear block_blue)
        (clear block_blue2)
        (clear block_blue3)
        (handempty robot)

        (ori_of block_r o8)
        (ori_of block_red1 o8)
        (ori_of block_blue o5)
        (ori_of block_blue2 o24)
        (ori_of block_blue3 o4)

        (gripper_at_init)
    )
    (:goal (and
        (on block_r block_red1)
        (ori_of block_r o2)
    ))
)
