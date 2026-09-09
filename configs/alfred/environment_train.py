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
        0: "pick_and_place_simple-Pencil-None-Desk",            # 25
        1: "pick_and_place_simple-CellPhone-None-Desk",         # 29
        2: "pick_and_place_simple-Book-None-Bed",               # 16
        3: "pick_and_place_simple-CellPhone-None-Bed",          # 25
        4: "pick_and_place_simple-KeyChain-None-ArmChair",      # 13
        5: "pick_two_obj_and_place-Bowl-None-Desk",             # 8
        6: "pick_two_obj_and_place-Laptop-None-Desk",           # 19
        7: "pick_two_obj_and_place-CellPhone-None-Bed",         # 64
        8: "pick_two_obj_and_place-Book-None-Bed",              # 47
        9: "pick_two_obj_and_place-CreditCard-None-ArmChair",   # 8
    },
    "set_2": { # clean & pick
        0: "pick_clean_then_place_in_recep-SoapBar-None-GarbageCan",    # 56
        1: "pick_clean_then_place_in_recep-SoapBar-None-Cart",          # 26
        2: "pick_clean_then_place_in_recep-Cloth-None-Cart",            # 11
        3: "pick_clean_then_place_in_recep-DishSponge-None-Shelf",      # 8
        4: "pick_clean_then_place_in_recep-Cloth-None-Shelf",           # 11
        5: "pick_and_place_simple-Candle-None-Cart",                    # 11
        6: "pick_and_place_simple-TissueBox-None-GarbageCan",           # 8
        7: "pick_two_obj_and_place-SoapBar-None-GarbageCan",            # 8
        8: "pick_two_obj_and_place-SprayBottle-None-Cart",              # 8
        9: "pick_two_obj_and_place-Candle-None-Shelf",                  # 8
    },
    "set_3": { # pick & pick two
        0: "pick_and_place_simple-CreditCard-None-Sofa",        # 17
        1: "pick_and_place_simple-RemoteControl-None-Sofa",     # 17
        2: "pick_and_place_simple-Pillow-None-ArmChair",        # 20
        3: "pick_and_place_simple-Newspaper-None-ArmChair",     # 8
        4: "pick_and_place_simple-TissueBox-None-CoffeeTable",  # 8
        5: "pick_two_obj_and_place-Newspaper-None-Sofa",        # 43
        6: "pick_two_obj_and_place-KeyChain-None-Sofa",         # 15
        7: "pick_two_obj_and_place-CellPhone-None-ArmChair",    # 8
        8: "pick_two_obj_and_place-CreditCard-None-ArmChair",   # 74
        9: "pick_two_obj_and_place-Plate-None-CoffeeTable",     # 13
    },
    "set_4": { # heat & clean
        0: "pick_heat_then_place_in_recep-Egg-None-DiningTable",        # 9
        1: "pick_heat_then_place_in_recep-Tomato-None-DiningTable",     # 10
        2: "pick_heat_then_place_in_recep-Apple-None-DiningTable",      # 10
        3: "pick_heat_then_place_in_recep-Mug-None-CoffeeMachine",      # 51
        4: "pick_heat_then_place_in_recep-Potato-None-GarbageCan",      # 15
        5: "pick_clean_then_place_in_recep-Spatula-None-DiningTable",   # 14
        6: "pick_clean_then_place_in_recep-Fork-None-DiningTable",      # 8
        7: "pick_clean_then_place_in_recep-ButterKnife-None-CounterTop",# 9
        8: "pick_clean_then_place_in_recep-Kettle-None-StoveBurner",    # 10
        9: "pick_clean_then_place_in_recep-Pan-None-StoveBurner",       # 10
    }, 
}

sequences = {
    "sequence_1": ["set_1", "set_2", "set_3", "set_4"], # default
    "sequence_2": ["set_2", "set_1", "set_4", "set_3"],
    "sequence_3": ["set_3", "set_2", "set_1", "set_4"],
    "sequence_4": ["set_4", "set_2", "set_3", "set_1"],
}
