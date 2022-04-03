(define (problem task)
	(:domain blocksworld)
	(:objects object0 object1 - obj)
	(:init
		(hand_empty)
		(on_table object0)
		(on_table object1)
		(clear object0)
		(clear object1)
		
	)
	(:goal (and 
		(on object1 object0))
	)
)