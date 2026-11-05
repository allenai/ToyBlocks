(define (problem rotate_block)
    (:domain blocks-tamp)
    (:objects
    block_r - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_r workspace_table)
        (clear block_r)
        (handempty robot)
        (gripper_at_init)

        (ori_of block_r o1)
    )
    (:goal (and
        (ori_of block_r o2)
    ))
)