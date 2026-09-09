rooms = {
    "set_1": {
        3: "Bedroom",
    },  
    "set_2": {
        4: "Bathroom",
    },   
    "set_3": {
        2: "Livingroom",
    },   
    "set_4": {
        1: "Kitchen",
    },
}

room_types = {
    "Kitchen": range(1, 31),
    "Livingroom": range(201, 231),
    "Bedroom": range(301, 331),
    "Bathroom": range(401, 431),
}

tasks = {
    "set_1": { # pick & pick two
        0: "pick_and_place_simple-CellPhone-None-ArmChair",     # 14
        1: "pick_and_place_simple-CreditCard-None-Shelf",       # 17
        2: "pick_two_obj_and_place-Pencil-None-Desk",           # 36
        3: "pick_two_obj_and_place-Laptop-None-Bed",            # 8
        4: "pick_two_obj_and_place-Pillow-None-ArmChair",       # 9
    },
    "set_2": { # clean & pick
        0: "pick_clean_then_place_in_recep-SoapBar-None-Shelf",     # 8
        1: "pick_clean_then_place_in_recep-DishSponge-None-Toilet", # 8
        2: "pick_and_place_simple-Candle-None-SideTable",           # 8
        3: "pick_two_obj_and_place-DishSponge-None-Cart",           # 11
        4: "pick_two_obj_and_place-SoapBar-None-GarbageCan",        # 8
    },
    "set_3": { # pick & pick two
        0: "pick_and_place_simple-Pillow-None-Sofa",                # 13
        1: "pick_and_place_simple-Box-None-ArmChair",               # 21
        2: "pick_two_obj_and_place-RemoteControl-None-CoffeeTable", # 21
        3: "pick_two_obj_and_place-CreditCard-None-Sofa",           # 16
        4: "pick_two_obj_and_place-Laptop-None-Ottoman",            # 10
    },
    "set_4": { # heat & clean
        0: "pick_heat_then_place_in_recep-Egg-None-GarbageCan",     # 13
        1: "pick_heat_then_place_in_recep-Tomato-None-SideTable",   # 9
        2: "pick_clean_then_place_in_recep-Mug-None-CoffeeMachine", # 32
        3: "pick_clean_then_place_in_recep-Pot-None-StoveBurner",   # 9
        4: "pick_clean_then_place_in_recep-Spoon-None-DiningTable", # 9
    }, 
}

sequences = {
    "sequence_1": ["set_1", "set_2", "set_3", "set_4"], # default
    "sequence_2": ["set_2", "set_1", "set_4", "set_3"],
    "sequence_3": ["set_3", "set_2", "set_1", "set_4"],
    "sequence_4": ["set_4", "set_2", "set_3", "set_1"],
}
