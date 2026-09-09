rooms = {
    "set_1": {
        #12: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        24: [["kitchen", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        26: [["kitchen", "bathroom"], ["bedroom", "livingroom"], ["livingroom", "kitchen"]],
    },
    "set_2": {
        #24: [["kitchen", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        12: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        26: [["kitchen", "bathroom"], ["bedroom", "livingroom"], ["livingroom", "kitchen"]],
    },
    "set_3": {
        #13: [["bedroom", "kitchen"], ["kitchen", "livingroom"], ["livingroom", "bathroom"]],
        12: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        24: [["kitchen", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
    },
    "set_4": {
        #26: [["kitchen", "bathroom"], ["bedroom", "livingroom"], ["livingroom", "kitchen"]],
        12: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        24: [["kitchen", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
    }
}

available_tasks = {
    12: [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 26, 27, 28, 29, 30], 
    24: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30], 
    13: [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 14, 15, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30],
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
