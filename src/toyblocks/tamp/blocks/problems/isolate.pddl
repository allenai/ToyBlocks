(define (problem isolate_block)
    (:domain blocks-tamp)
    (:objects
    block_r block_red1 - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_r workspace_table)
        (clear block_r)

        (on_table block_red1 workspace_table)
        (clear block_red1)

        (next_to block_r block_red1)
        (pick_obstructs block_red1 block_r)

        (ori_of block_r o9)
        (ori_of block_red1 o2)

        (handempty robot)
        (gripper_at_init)

    )
    (:goal (and
        (clear block_r)
        (not (pick_obstructs block_red1 block_r))
    ))
    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
    ;friction_range = [0.1, 0.4]
)
