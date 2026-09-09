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
        #3: "Bedroom",
        4: "Bathroom",
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

tasks = {
    "set_1": {
    },
    "set_2": {
        0: "pick_heat_then_place_in_recep-Cup-None-Shelf",          # 35
        1: "pick_heat_then_place_in_recep-Potato-None-CounterTop",  # 35
        2: "pick_heat_then_place_in_recep-Bread-None-DiningTable",  # 8
        3: "pick_heat_then_place_in_recep-Apple-None-DiningTable",  # 23
        4: "pick_heat_then_place_in_recep-Apple-None-GarbageCan",   # 32
        #3: "pick_heat_then_place_in_recep-Mug-None-SideTable",      # 17
        #4: "pick_heat_then_place_in_recep-Tomato-None-GarbageCan",  # 20
    },
    "set_3": {
        0: "pick_two_obj_and_place-CellPhone-None-Sofa",            # 12
        1: "pick_two_obj_and_place-KeyChain-None-ArmChair",         # 14
        2: "pick_two_obj_and_place-RemoteControl-None-Ottoman",     # 19
        3: "pick_two_obj_and_place-Laptop-None-Desk",               # 20
        4: "pick_two_obj_and_place-Book-None-Bed",                  # 48
    },
    "set_4": {
    },
    "set_5": {
        0: "pick_and_place_simple-Laptop-None-Ottoman",             # 17
        1: "pick_and_place_simple-CellPhone-None-ArmChair",         # 14
        2: "pick_and_place_simple-KeyChain-None-Safe",              # 8
        3: "pick_and_place_simple-Cloth-None-CounterTop",           # 8
        4: "pick_and_place_simple-Pencil-None-Desk",                # 26
        #4: "pick_and_place_simple-CreditCard-None-CoffeeTable",     # 8
        #2: "pick_and_place_simple-Laptop-None-Ottoman",             # 17
        #4: "pick_and_place_simple-Laptop-None-Bed",                 # 8
    },
    "set_6": {
        0: "pick_clean_then_place_in_recep-Spoon-None-CounterTop",      # 8
        1: "pick_clean_then_place_in_recep-Fork-None-DiningTable",      # 14
        2: "pick_clean_then_place_in_recep-Plate-None-Shelf",           # 29
        3: "pick_clean_then_place_in_recep-Cloth-None-Cart",            # 11
        4: "pick_clean_then_place_in_recep-DishSponge-None-Toilet",    # 8
        #2: "pick_clean_then_place_in_recep-Spatula-None-SideTable",     # 8
        #3: "pick_clean_then_place_in_recep-Cloth-None-Shelf",           # 8
        #4: "pick_clean_then_place_in_recep-DishSponge-None-CounterTop", # 11
    },
}

sequences = {
    "sequence_1": ["set_2", "set_3", "set_5", "set_6"],
    "sequence_2": ["set_3", "set_6", "set_2", "set_5"],
    "sequence_3": ["set_5", "set_3", "set_6", "set_2"],
    "sequence_4": ["set_6", "set_2", "set_3", "set_5"],
}
