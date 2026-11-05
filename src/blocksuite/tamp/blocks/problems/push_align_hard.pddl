(define (problem Push-Align-Hard)
    (:domain blocks-tamp)
    (:objects
    block_r - block_r_e
    block_o1 - block_o_b
    block_red1 block_green block_blue - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_r workspace_table)
        (on block_o1 block_green)
        (on_table block_red1 workspace_table)
        (on_table block_green workspace_table)
        (on_table block_blue workspace_table)
        (clear block_r)
        (clear block_o1)
        (clear block_red1)
        (clear block_blue)
        (handempty robot)

        (ori_of block_r o1)
        (ori_of block_o1 o19)
        (ori_of block_red1 o1)
        (ori_of block_green o2)
        (ori_of block_blue o1)

        (gripper_at_init)
    )
    (:goal (and
        (on_right block_r block_red1)
    ))
    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
    ;friction_range = [0.1, 0.4]
)