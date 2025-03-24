rosservice call /execute_trajectory
rosservice call /head_scan_service
rosservice call /picker_service '{"object_class": "book"}'
rosservice call /deliver_object_service
rosservice call /handover_object_service
