(define (problem letter-color-10-20260902_211523_512636_5048)
    (:domain blocks-tamp)
    (:objects
    block_a block_i - block_letter
    block_2 - block_digit
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_a workspace_table)
        (on_table block_i workspace_table)
        (on_table block_2 workspace_table)
        (clear block_a)
        (clear block_i)
        (clear block_2)
        (handempty robot)

        (ori_of block_a o3)
        (ori_of block_i o12)
        (ori_of block_2 o6)

        (gripper_at_init)
    )
    (:goal (and
        (on block_a block_i)
        (on block_i block_2)
        (ori_of block_a o2)
        (ori_of block_i o2)
        (ori_of block_2 o2)
    ))
)