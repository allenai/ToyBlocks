(define (problem red_on_green_2block)
    (:domain blocks-tamp)
    (:objects
    block_red1 block_green1 - block
    workspace_table - table
    robot - gripper
    )
    (:init
        (on_table block_red1 workspace_table)
        (clear block_red1)

        (on_table block_green1 workspace_table)
        (clear block_green1)

        (handempty robot)
        (gripper_at_init)

        (ori_of block_red1 o1)
        (ori_of block_green1 o1)
    )
    (:goal (and
        (on block_red1 block_green1)
    ))
)
