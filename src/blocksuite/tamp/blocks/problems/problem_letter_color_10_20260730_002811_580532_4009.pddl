(define (problem letter-color-10-20260730_002811_580532_4009)
    (:domain blocks-tamp)
    (:objects
    block_e block_c block_s - block_letter
    block_red5 block_red1 - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_e workspace_table)
        (on block_c block_e)
        (on_table block_s workspace_table)
        (on_table block_red5 workspace_table)
        (on_table block_red1 workspace_table)
        (clear block_c)
        (clear block_s)
        (clear block_red5)
        (clear block_red1)
        (handempty robot)

        (ori_of block_e o22)
        (ori_of block_c o22)
        (ori_of block_s o18)
        (ori_of block_red5 o5)
        (ori_of block_red1 o21)

        (gripper_at_init)
    )
    (:goal (and
        (on_table block_red1 workspace_table)
        (on block_red5 block_c)
        (on_right block_s block_c)
        (ori_of block_e o15)
        (on_right block_e block_red1)
    ))

    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0.0, 0.04]
    ;friction_range = [0.1, 0.4]
)