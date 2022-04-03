(define (domain blocksworld)
    (:requirements :strips :equality)
    (:types 
        obj ; objects
    )
    (:predicates 
        (clear ?x - obj)
        (on_table ?x - obj)
        (hand_empty)
        (holding ?x - obj)
        (on ?x ?y - obj)
    )


    (:action pickup
        :parameters (?ob - obj)
        :precondition (and 
            (clear ?ob) 
            (on_table ?ob) 
            (hand_empty)
        )
        :effect (and 
            (holding ?ob)
            (not (clear ?ob))
            (not (on_table ?ob)) 
            (not (hand_empty))
        )
    )

    (:action putdown
        :parameters (?ob - obj)
        :precondition (and 
            (holding ?ob)
        )
        :effect (and
            (clear ?ob)
            (hand_empty)
            (on_table ?ob) 
            (not (holding ?ob))
        )
    )

    (:action stack
        :parameters (?ob - obj ?underob - obj)
        :precondition (and 
            (clear ?underob) 
            (holding ?ob)
        )
        :effect (and 
            (hand_empty) 
            (clear ?ob) 
            (on ?ob ?underob)
            (not (clear ?underob)) 
            (not (holding ?ob))
        )
    )

    (:action unstack
        :parameters (?ob - obj ?underob - obj)
        :precondition (and 
            (on ?ob ?underob) 
            (clear ?ob) 
            (hand_empty)
        )
        :effect (and 
            (holding ?ob) 
            (clear ?underob)
            (not (on ?ob ?underob)) 
            (not (clear ?ob)) 
            (not (hand_empty))
        )
    )
)