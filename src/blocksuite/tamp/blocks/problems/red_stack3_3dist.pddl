(define (problem red_stack3_3dist)
    (:domain blocks-tamp)
    (:objects
    block_red1 block_red2 block_red3 block_red4 - block
    block_green1 - block
    block_blue1 - block
    block_yellow1 - block
    workspace_table - table
    robot - gripper
    )
    (:init
        (on_table block_red1 workspace_table)
        (on_table block_red2 workspace_table)
        (on_table block_red3 workspace_table)
        (on_table block_red4 workspace_table)
        (on_table block_green1 workspace_table)
        (on_table block_blue1 workspace_table)
        (on_table block_yellow1 workspace_table)

        (clear block_red1)
        (clear block_red2)
        (clear block_red3)
        (clear block_red4)
        (clear block_green1)
        (clear block_blue1)
        (clear block_yellow1)

        (handempty robot)
        (gripper_at_init)

        (ori_of block_red1 o8)
        (ori_of block_red2 o8)
        (ori_of block_red3 o8)
        (ori_of block_red4 o8)
        (ori_of block_green1 o8)
        (ori_of block_blue1 o8)
        (ori_of block_yellow1 o8)
    )
    (:goal (and
        (on block_red2 block_red1)
        (on block_red3 block_red2)
    ))
)
