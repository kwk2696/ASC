rooms = {
    "set_1": {
        2: "Livingroom",
        3: "Bedroom",
    },  
    "set_2": {
        0: "Kitchen",
    },   
    "set_3": {
        #0: "Kitchen",
        2: "Livingroom",
        3: "Bedroom",
        #4: "Bathroom",
    },   
    "set_4": {
        0: "Kitchen",
    },
    "set_5": {
        #0: "Kitchen",
        2: "Livingroom",
        3: "Bedroom",
        #4: "Bathroom",
    },
    "set_6": {
        0: "Kitchen",
        4: "Bathroom",
    },
}

room_types = {
    "Kitchen": range(1, 31),
    "Livingroom": range(201, 231),
    "Bedroom": range(301, 331),
    "Bathroom": range(401, 431),
}
        
# ban list : apple, cabinet, toilet
# set1 & set5 is not used
tasks = {
    "set_1": {
        0: "look_at_obj_in_light-Newspaper-None-DeskLamp", # 200
        1: "look_at_obj_in_light-Vase-None-DeskLamp",
        2: "look_at_obj_in_light-Watch-None-DeskLamp",
        3: "look_at_obj_in_light-Plate-None-DeskLamp",
        4: "look_at_obj_in_light-AlarmClock-None-DeskLamp", # 300
        5: "look_at_obj_in_light-CellPhone-None-DeskLamp",
        6: "look_at_obj_in_light-KeyChain-None-DeskLamp",
        7: "look_at_obj_in_light-CreditCard-None-DeskLamp",
    },
    "set_2": {
        0: "pick_heat_then_place_in_recep-Mug-None-Shelf",          # 35
        1: "pick_heat_then_place_in_recep-Plate-None-Shelf",        # 8
        2: "pick_heat_then_place_in_recep-Tomato-None-CounterTop",  # 32
        3: "pick_heat_then_place_in_recep-Egg-None-CounterTop",     # 29
        4: "pick_heat_then_place_in_recep-Egg-None-DiningTable",    # 29
        5: "pick_heat_then_place_in_recep-Potato-None-DiningTable", # 20
        6: "pick_heat_then_place_in_recep-Tomato-None-SideTable",   # 29
        7: "pick_heat_then_place_in_recep-Cup-None-SideTable",      # 11
        8: "pick_heat_then_place_in_recep-Potato-None-GarbageCan",  # 52
        9: "pick_heat_then_place_in_recep-Bread-None-GarbageCan",   # 8
    },
    "set_3": {
        0: "pick_two_obj_and_place-RemoteControl-None-Sofa",        # 35
        1: "pick_two_obj_and_place-CreditCard-None-Sofa",           # 17
        2: "pick_two_obj_and_place-Newspaper-None-ArmChair",        # 36
        3: "pick_two_obj_and_place-CellPhone-None-ArmChair",        # 11
        4: "pick_two_obj_and_place-RemoteControl-None-Ottoman",     # 19
        5: "pick_two_obj_and_place-Laptop-None-Ottoman",            # 12
        6: "pick_two_obj_and_place-Pencil-None-Desk",               # 39
        7: "pick_two_obj_and_place-AlarmClock-None-Desk",           # 35
        8: "pick_two_obj_and_place-Laptop-None-Bed",                # 9
        9: "pick_two_obj_and_place-CellPhone-None-Bed",             # 70
    },
    "set_4": {
        0: "pick_cool_then_place_in_recep-Tomato-None-CounterTop", # 1
        1: "pick_cool_then_place_in_recep-Bread-None-CounterTop",
        2: "pick_cool_then_place_in_recep-Bread-None-DiningTable",
        3: "pick_cool_then_place_in_recep-Lettuce-None-DiningTable",
        4: "pick_cool_then_place_in_recep-Potato-None-SideTable",
        5: "pick_cool_then_place_in_recep-Apple-None-SideTable",
        6: "pick_cool_then_place_in_recep-Potato-None-GarbageCan",
        7: "pick_cool_then_place_in_recep-Tomato-None-GarbageCan",
        8: "pick_cool_then_place_in_recep-Mug-None-Shelf",
        9: "pick_cool_then_place_in_recep-Pot-None-Shelf",
    },
    "set_5": {
        0: "pick_and_place_simple-KeyChain-None-Sofa",              # 35
        1: "pick_and_place_simple-Laptop-None-Sofa",                # 8
        2: "pick_and_place_simple-RemoteControl-None-ArmChair",     # 44
        3: "pick_and_place_simple-KeyChain-None-ArmChair",          # 26
        4: "pick_and_place_simple-Newspaper-None-Ottoman",          # 23
        5: "pick_and_place_simple-Pillow-None-Ottoman",             # 17
        6: "pick_and_place_simple-Laptop-None-Desk",                # 8
        7: "pick_and_place_simple-Pen-None-Desk",                   # 8
        8: "pick_and_place_simple-CellPhone-None-Bed",              # 32
        9: "pick_and_place_simple-Book-None-Bed",                   # 17
    },
    "set_6": {
        0: "pick_clean_then_place_in_recep-Spatula-None-CounterTop",    # 31
        1: "pick_clean_then_place_in_recep-Knife-None-CounterTop",      # 17
        2: "pick_clean_then_place_in_recep-Kettle-None-DiningTable",    # 23
        3: "pick_clean_then_place_in_recep-Spoon-None-DiningTable",     # 26
        4: "pick_clean_then_place_in_recep-Fork-None-SideTable",        # 14
        5: "pick_clean_then_place_in_recep-Knife-None-SideTable",       # 14
        6: "pick_clean_then_place_in_recep-DishSponge-None-Shelf",      # 23
        7: "pick_clean_then_place_in_recep-SoapBar-None-Shelf",         # 8   
        8: "pick_clean_then_place_in_recep-SoapBar-None-CounterTop",    # 50
        9: "pick_clean_then_place_in_recep-Cloth-None-CounterTop",      # 20

    },
}

sequences = {
    "sequence_1": ["set_2", "set_3", "set_5", "set_6"],
    "sequence_2": ["set_3", "set_6", "set_2", "set_5"],
    "sequence_3": ["set_5", "set_3", "set_6", "set_2"],
    "sequence_4": ["set_6", "set_2", "set_3", "set_5"],
}
