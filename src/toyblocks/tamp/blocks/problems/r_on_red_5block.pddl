(define (problem r_on_red_5block)
    (:domain blocks-tamp)
    (:objects
    block_r - block_r_e
    block_red1 - block
    block_blue1 block_blue2 block_blue3 - block
    workspace_table - table
    robot - gripper
    )
    (:init
        (on_table block_r workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_blue1 workspace_table)
        (on_table block_blue2 workspace_table)
        (on_table block_blue3 workspace_table)
        (clear block_r)
        (clear block_red1)
        (clear block_blue1)
        (clear block_blue2)
        (clear block_blue3)
        (handempty robot)
        (gripper_at_init)

        (ori_of block_r o8)
        (ori_of block_red1 o8)
        (ori_of block_blue1 o5)
        (ori_of block_blue2 o24)
        (ori_of block_blue3 o4)
    )
    (:goal (and
        (on block_r block_red1)
        (ori_of block_r o2)
    ))
)
