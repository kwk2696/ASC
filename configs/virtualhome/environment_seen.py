rooms = {
    "set_1": {
        20: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
    },
    "set_2": {
        1: [["kitchen", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
    },
    "set_3": {
        34: [["bedroom", "kitchen"], ["livingroom", "kitchen"], ["livingroom", "bathroom"]],
    },
    "set_4": {
        26: [["kitchen", "bathroom"], ["bedroom", "livingroom"], ["livingroom", "kitchen"]],
    }
}

available_tasks = {
    20: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 26, 27, 28, 29, 30],
    1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30], 
    34: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 24, 27, 28, 29, 30], 
    26: [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 28, 29, 30], 
}

tasks = {
    "set_1": {
        0: "Turn on tv", #
        1: "Turn on radio", 
        2: "Turn on microwave",
        3: "Turn on stove",
        4: "Turn on computer",
        5: "Open cabinet", #
        7: "Open microwave",
        8: "Open stove",
        9: "Open fridge",
        24: "Open toilet",
        10: "Place apple on sofa", # 
        11: "Place book on coffeetable", 
        12: "Place mug on coffeetable",
        13: "Place towel on sofa",
        14: "Place paper on bed", 
        15: "Put mug in microwave", #
        17: "Put towel in closet",
        18: "Put apple in fridge",
        19: "Put paper in bookshelf",
        29: "Put bananas in fridge",
    },
    "set_2": {
        0: "Turn on tv", #
        1: "Turn on radio", 
        3: "Turn on stove",
        20: "Turn on coffeemaker",
        21: "Turn on dishwasher",
        6: "Open dishwasher", # 
        7: "Open microwave",
        8: "Open stove",
        23: "Open coffeepot",
        24: "Open toilet",
        10: "Place apple on sofa", #
        14: "Place paper on bed",
        26: "Place chips on sofa",
        27: "Place book on desk",
        28: "Place bananas on bed",
        15: "Put mug in microwave", #
        16: "Put plate in dishwasher",
        18: "Put apple in fridge",
        29: "Put bananas in fridge",
        30: "Put book in closet",
    },
    "set_3": {
        0: "Turn on tv", # 
        2: "Turn on microwave",
        3: "Turn on stove",
        4: "Turn on computer",
        21: "Turn on dishwasher",
        5: "Open cabinet", #
        6: "Open dishwasher", 
        7: "Open microwave",
        8: "Open stove",
        9: "Open fridge",
        11: "Place book on coffeetable", # 
        12: "Place mug on coffeetable",
        13: "Place towel on sofa",
        14: "Place paper on bed", 
        27: "Place book on desk",
        15: "Put mug in microwave", #
        16: "Put plate in dishwasher",
        17: "Put towel in closet",
        18: "Put apple in fridge",
        30: "Put book in closet",
    },
    "set_4": {
        0: "Turn on tv", # 
        1: "Turn on radio", 
        2: "Turn on microwave",
        3: "Turn on stove",
        20: "Turn on coffeemaker",
        5: "Open cabinet", #
        6: "Open dishwasher", 
        9: "Open fridge",
        23: "Open coffeepot",
        24: "Open toilet",
        10: "Place apple on sofa", # 
        11: "Place book on coffeetable", 
        12: "Place mug on coffeetable",
        13: "Place towel on sofa",
        28: "Place bananas on bed",
        15: "Put mug in microwave", #
        16: "Put plate in dishwasher",
        17: "Put towel in closet",
        18: "Put apple in fridge",
        29: "Put bananas in fridge",
    },
}

sequences = {
    "sequence_1": ["set_1", "set_2", "set_3", "set_4"], # default
    "sequence_2": ["set_2", "set_1", "set_4", "set_3"],
    "sequence_3": ["set_3", "set_2", "set_1", "set_4"],
    "sequence_4": ["set_4", "set_2", "set_3", "set_1"],
}
