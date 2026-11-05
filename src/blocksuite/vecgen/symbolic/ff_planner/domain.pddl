(define (domain blocks-tamp)
  (:requirements :strips :typing :adl :conditional-effects :quantified-preconditions)
  (:types 
    block
    table  
    gripper
    conf
    pose
    traj
    dir
    rot_dir flip_dir - dir 
    align 
    ori
    face
  )

  (:predicates
    (at ?b - block ?p - pose)
    (near ?b - block ?p - pose)
    (holding ?g - gripper ?b - block)
    (clear ?b - block)
    (handempty ?g - gripper)
    (collides ?b1 - block ?p - pose ?t - traj)
    (on_table ?b - block ?t - table)
    (valid_grasp ?g - gripper ?b - block ?gp - pose)
    (on ?b1 - block ?b2 - block)
    (valid_right_pose ?b1 - block ?b2 - block ?p - pose)
    (on_right ?b1 - block ?b2 - block)
    (ori_of ?b - block ?o - ori)
    (up_face ?o - ori ?u - face)
    (front_face ?o - ori ?f - face)
    (top_align ?o - ori ?a - align)
    (yaw_next ?d - rot_dir ?o - ori ?o2 - ori)
    (flip_next ?d - flip_dir ?o - ori ?o2 - ori)  
  )

  (:action move_from_block_to_block
    :parameters (?g - gripper ?b1 - block ?b2 - block ?b3 - block ?cp - pose ?gp - pose ?pp - pose ?t - traj)
    :precondition (and 
      (on ?b1 ?b2)
      (clear ?b1)
      (at ?b1 ?cp)
      (handempty ?g)
      (valid_grasp ?g ?b1 ?gp)
      (forall (?b4 - block) (not (collides ?b4 ?gp ?t)))
      (clear ?b3)
      (not (= ?b2 ?b3))
      (not (= ?b1 ?b2))
      (not (= ?b1 ?b3))
      (forall (?b5 - block) (not (collides ?b5 ?pp ?t)))
    )
    :effect (and 
      (at ?b1 ?pp)
      (on ?b1 ?b3)
      (clear ?b1)
      (clear ?b2)
      (not (clear ?b3))
      (not (at ?b1 ?cp))
      (not (on ?b1 ?b2))
      (forall (?p - pose) (forall (?t1 - traj) (not (collides ?b1 ?p ?t1))))
    )
  )

  (:action move_from_table_to_block
    :parameters (?g - gripper ?b1 - block ?b2 - block ?table - table ?cp - pose ?gp - pose ?t - traj)
    :precondition (and 
      (on_table ?b1 ?table)
      (clear ?b1)
      (at ?b1 ?cp)
      (handempty ?g)
      (valid_grasp ?g ?b1 ?gp)
      (forall (?b4 - block) (not (collides ?b4 ?gp ?t)))
      (clear ?b2)
      (not (= ?b1 ?b2))
    )
    :effect (and 
      (not (at ?b1 ?cp))
      (on ?b1 ?b2)
      (clear ?b1)
      (not (clear ?b2))
      (not (on_table ?b1 ?table))
      (forall (?p - pose) (forall (?t1 - traj) (not (collides ?b1 ?p ?t1))))
    )
  )

  (:action move_from_block_to_table
    :parameters (?g - gripper ?b1 - block ?b2 - block ?table - table ?cp - pose ?gp - pose ?pp - pose ?t - traj)
    :precondition (and 
      (on ?b1 ?b2)
      (clear ?b1)
      (at ?b1 ?cp)
      (handempty ?g)
      (valid_grasp ?g ?b1 ?gp)
      (forall (?b4 - block) (not (collides ?b4 ?gp ?t)))
      (forall (?b5 - block) (not (collides ?b5 ?pp ?t)))
    )
    :effect (and 
      (at ?b1 ?pp)
      (on_table ?b1 ?table)
      (clear ?b1)
      (clear ?b2)
      (not (at ?b1 ?cp))
      (not (on ?b1 ?b2))
      (forall (?p - pose) (forall (?t1 - traj) (not (collides ?b1 ?p ?t1))))
    )
  )

  (:action move_from_block_to_right
  :parameters (?g - gripper ?b1 - block ?b2 - block ?b3 - block ?table - table ?cp - pose ?gp - pose ?pp - pose ?t - traj)
    :precondition (and 
      (on ?b1 ?b2)
      (on_table ?b3 ?table)
      (clear ?b1)
      (at ?b1 ?cp)
      (handempty ?g)
      (valid_grasp ?g ?b1 ?gp)
      (forall (?b4 - block) (not (collides ?b4 ?gp ?t)))
      (forall (?b5 - block) (not (collides ?b5 ?pp ?t)))
    )
    :effect (and 
      (at ?b1 ?pp)
      (on_table ?b1 ?table)
      (clear ?b1)
      (clear ?b2)
      (not (on ?b1 ?b2))
      (on_right ?b1 ?b3)
      (forall (?b4 - block) (not (on_right ?b1 ?b4)))
      (not (at ?b1 ?cp))
      (forall (?p - pose) (forall (?t1 - traj) (not (collides ?b1 ?p ?t1))))
    )
  )

  (:action push_to_pose
    :parameters (?g - gripper ?b1 - block ?table - table ?cp - pose ?np - pose ?t - traj)
    :precondition (and 
      (on_table ?b1 ?table)
      (at ?b1 ?cp)
      (clear ?b1)
    )
    :effect (and 
      (not (at ?b1 ?cp))
      (near ?b1 ?np)
    )
  
  )

  (:action push_to_right
    :parameters (?g - gripper ?b1 - block ?b2 - block ?t - table ?from - pose ?to - pose ?gp - pose ?traj - traj)
    :precondition (and 
      (on_table ?b1 ?t)
      (on_table ?b2 ?t)
      (handempty ?g)
      (at ?b1 ?from)
      (not (at ?b1 ?to))
      (valid_grasp ?g ?b1 ?gp)
      (forall (?b3 - block) (not (collides ?b3 ?gp ?traj)))
      (forall (?b4 - block) (not (collides ?b4 ?to ?traj)))
    )
    :effect (and 
        (at ?b1 ?to)
        (not (at ?b1 ?from))
        (on_right ?b1 ?b2)
        (forall (?b4 - block) (not (on_right ?b1 ?b4)))
        (forall (?p - pose) (forall (?traj2 - traj) (not (collides ?b1 ?p ?traj2))))
    )
  )

  (:action pick
    :parameters (?g - gripper ?b1 - block ?b2 - block ?p - pose ?gp - pose ?t - traj)
    :precondition (and 
      (on ?b1 ?b2)
      (clear ?b1)
      (at ?b1 ?p)
      (handempty ?g)
      (valid_grasp ?g ?b1 ?gp)
      (forall (?b3 - block) (not (collides ?b3 ?gp ?t)))
    )
    :effect (and 
      (holding ?g ?b1)
      (not (handempty ?g))
      (not (clear ?b2))
      (forall (?p2 - pose) (forall (?traj2 - traj) (not (collides ?b1 ?p2 ?traj2))))
      (clear ?b2)
      (not (on ?b1 ?b2))
      (not (at ?b1 ?p))
    )
  )

  (:action pick_from_table
    :parameters (?g - gripper ?b1 - block ?t - table ?p - pose ?gp - pose ?traj - traj)
    :precondition (and 
      (on_table ?b1 ?t)
      (clear ?b1)
      (handempty ?g)
      (forall (?b3 - block) (not (collides ?b3 ?gp ?traj)))
    )
    :effect (and 
      (holding ?g ?b1)
      (not (handempty ?g))
      (not (on_table ?b1 ?t))
      (not (at ?b1 ?p))
      (forall (?p2 - pose) (forall (?traj2 - traj) (not (collides ?b1 ?p2 ?traj2))))
    )
  )

  (:action rotate
    :parameters (?g - gripper ?b - block ?d - rot_dir ?o - ori ?o2 - ori ?table - table ?t - traj)
    :precondition (and 
      (handempty ?g)
      (on_table ?b ?table)
      (ori_of ?b ?o)
      (yaw_next ?d ?o ?o2)
    )
    :effect (and 
      (not (ori_of ?b ?o)) 
      (ori_of ?b ?o2)
    )
  )

  (:action nudge
    :parameters (?g - gripper ?b - block ?p - pose ?gp - pose ?t - traj)
    :precondition (and 
      (near ?b ?p)
      (not (at ?b ?p))
      (valid_grasp ?g ?b ?gp)
      (forall (?b2 - block) (not (collides ?b2 ?gp ?t)))
      (forall (?b2 - block) (not (collides ?b2 ?p ?t)))
      (handempty ?g)
    )
    :effect (and
      (not (near ?b ?p))
      (at ?b ?p)
    )
  )

  (:action reorient
    :parameters (?g - gripper ?b - block ?d - flip_dir ?o - ori ?o2 - ori ?table - table ?t - traj)
    :precondition (and
      (handempty ?g)
      (on_table ?b ?table)
      (ori_of ?b ?o)
      (flip_next ?d ?o ?o2)
    )
    :effect (and
      (not (ori_of ?b ?o))
      (ori_of ?b ?o2)
    )
  )

)
