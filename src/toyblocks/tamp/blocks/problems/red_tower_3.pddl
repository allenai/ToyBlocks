(define (problem red_tower_3)
    (:domain blocks-tamp)
    (:objects
    block_red1 block_red2 block_red3 - block
    workspace_table - table
    robot - gripper
    )
    (:init
        (on_table block_red1 workspace_table)
        (on_table block_red2 workspace_table)
        (on_table block_red3 workspace_table)
        (clear block_red1)
        (clear block_red2)
        (clear block_red3)
        (handempty robot)
        (gripper_at_init)

        (ori_of block_red1 o8)
        (ori_of block_red2 o8)
        (ori_of block_red3 o8)
    )
    (:goal (and
        (on block_red2 block_red1)
        (on block_red3 block_red2)
    ))
)
