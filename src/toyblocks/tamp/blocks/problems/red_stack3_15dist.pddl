(define (problem red_stack3_15dist)
    (:domain blocks-tamp)
    (:objects
    block_red1 block_red2 block_red3 block_red4 - block
    block_green1 block_green2 - block
    block_blue1 block_blue2 - block
    block_yellow1 block_yellow2 - block
    block_orange1 block_orange2 - block
    block_purple1 block_purple2 - block
    block_cyan1 block_cyan2 - block
    block_gray1 - block
    block_pink1 - block
    block_brown1 - block
    workspace_table - table
    robot - gripper
    )
    (:init
        (on_table block_red1 workspace_table)
        (on_table block_red2 workspace_table)
        (on_table block_red3 workspace_table)
        (on_table block_red4 workspace_table)
        (on_table block_green1 workspace_table)
        (on_table block_green2 workspace_table)
        (on_table block_blue1 workspace_table)
        (on_table block_blue2 workspace_table)
        (on_table block_yellow1 workspace_table)
        (on_table block_yellow2 workspace_table)
        (on_table block_orange1 workspace_table)
        (on_table block_orange2 workspace_table)
        (on_table block_purple1 workspace_table)
        (on_table block_purple2 workspace_table)
        (on_table block_cyan1 workspace_table)
        (on_table block_cyan2 workspace_table)
        (on_table block_gray1 workspace_table)
        (on_table block_pink1 workspace_table)
        (on_table block_brown1 workspace_table)

        (clear block_red1)
        (clear block_red2)
        (clear block_red3)
        (clear block_red4)
        (clear block_green1)
        (clear block_green2)
        (clear block_blue1)
        (clear block_blue2)
        (clear block_yellow1)
        (clear block_yellow2)
        (clear block_orange1)
        (clear block_orange2)
        (clear block_purple1)
        (clear block_purple2)
        (clear block_cyan1)
        (clear block_cyan2)
        (clear block_gray1)
        (clear block_pink1)
        (clear block_brown1)

        (handempty robot)
        (gripper_at_init)

        (ori_of block_red1 o8)
        (ori_of block_red2 o8)
        (ori_of block_red3 o8)
        (ori_of block_red4 o8)
        (ori_of block_green1 o8)
        (ori_of block_green2 o8)
        (ori_of block_blue1 o8)
        (ori_of block_blue2 o8)
        (ori_of block_yellow1 o8)
        (ori_of block_yellow2 o8)
        (ori_of block_orange1 o8)
        (ori_of block_orange2 o8)
        (ori_of block_purple1 o8)
        (ori_of block_purple2 o8)
        (ori_of block_cyan1 o8)
        (ori_of block_cyan2 o8)
        (ori_of block_gray1 o8)
        (ori_of block_pink1 o8)
        (ori_of block_brown1 o8)
    )
    (:goal (and
        (on block_red2 block_red1)
        (on block_red3 block_red2)
    ))
)
