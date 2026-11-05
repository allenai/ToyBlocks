(define (problem isolate_block)
    (:domain blocks-tamp)
    (:objects
    block_r block_red1 block_red2 block_red3 block_red4 block_red5 - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_r workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_red2 workspace_table)
        (on_table block_red3 workspace_table)
        (on_table block_red4 workspace_table)
        (on_table block_red5 workspace_table)
        (clear block_r)
        (clear block_red1)
        (clear block_red2)
        (clear block_red3)
        (clear block_red4)
        (clear block_red5)

        (on_right block_red1 block_r)
        (on_right block_red2 block_red1)
        (on_front block_red3 block_r)
        (on_front block_red4 block_red1)
        (on_front block_red5 block_red2)

        (pick_obstructs block_red1 block_r)

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
