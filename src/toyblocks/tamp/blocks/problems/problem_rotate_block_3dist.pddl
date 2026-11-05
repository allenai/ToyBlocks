(define (problem rotate_block_3dist)
    ; Variant of rotate_block.pddl that adds three distractor cuboids
    ; (block_red1 / block_green / block_blue), matching the scene used to
    ; record the apr28_rotate_3dist dataset.
    ; Goal is unchanged from rotate_block.pddl: rotate block_r from o1 to o2.
    (:domain blocks-tamp)
    (:objects
    block_r - block
    block_red1 block_green block_blue - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_r workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_green workspace_table)
        (on_table block_blue workspace_table)
        (clear block_r)
        (clear block_red1)
        (clear block_green)
        (clear block_blue)
        (handempty robot)
        (gripper_at_init)

        (ori_of block_r o1)
        (ori_of block_red1 o1)
        (ori_of block_green o1)
        (ori_of block_blue o1)
    )
    (:goal (and
        (ori_of block_r o2)
    ))
)
