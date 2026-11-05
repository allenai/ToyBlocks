(define (problem Push-or-Align)
    (:domain blocks-tamp)
    (:objects
    block_r block_blue - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_r workspace_table)
        (on_table block_blue workspace_table)
        (clear block_r)
        (clear block_blue)
        (handempty robot)

        (ori_of block_r o2)
        (ori_of block_blue o8)

        (gripper_at_init)
    )
    (:goal (and
        (next_to block_r block_blue)
        ; (on_right block_r block_blue)
    ))
    ;friction_range = [0.1, 0.4]
)