rooms = {
    "set_1": {
        7: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        8: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        9: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        #28: [["bedroom", "bathroom"], ["kitchen", "bedroom"], ["livingroom", "kitchen"]],
    },
    "set_2": {
        7: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        8: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        9: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        #28: [["bedroom", "bathroom"], ["kitchen", "bedroom"], ["livingroom", "kitchen"]],
    },
    "set_3": {
        7: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        8: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        9: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        #28: [["bedroom", "bathroom"], ["kitchen", "bedroom"], ["livingroom", "kitchen"]],
    },
    "set_4": {
        7: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        8: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        9: [["bedroom", "bathroom"], ["bedroom", "kitchen"], ["livingroom", "kitchen"]],
        #28: [["bedroom", "bathroom"], ["kitchen", "bedroom"], ["livingroom", "kitchen"]],
    }
}

available_tasks = {
    7: [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 27, 29, 30], 
    8: [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 26, 27, 29, 30], 
    9: [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 26, 27, 29, 30], 
    28: [0, 1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 26, 27, 29, 30], 
}

tasks = {
    "set_1": {
        0: "Turn on tv", 
        1: "Turn on radio", 
        #2: "Turn on microwave",
        3: "Turn on stove",
        4: "Turn on computer",
        #20: "Turn on coffeemaker",
        21: "Turn on dishwasher",
    },
    "set_2": {
        5: "Open cabinet",
        6: "Open dishwasher", 
        #7: "Open microwave",
        8: "Open stove",
        9: "Open fridge",
        #23: "Open coffeepot",
        24: "Open toilet",
    },
    "set_3": {
        10: "Place apple on sofa", 
        11: "Place book on coffeetable", 
        #12: "Place mug on coffeetable",
        13: "Place towel on sofa",
        14: "Place paper on bed",
        #26: "Place chips on sofa",
        27: "Place book on desk",
    },
    "set_4": {
        15: "Put mug in microwave",
        #16: "Put plate in dishwasher",
        17: "Put towel in closet",
        18: "Put apple in fridge",
        19: "Put paper in bookshelf",
        #29: "Put bananas in fridge",
        30: "Put book in closet",
    }
}

sequences = {
    "sequence_1": ["set_1", "set_2", "set_3", "set_4"], # default
    "sequence_2": ["set_2", "set_1", "set_4", "set_3"],
    "sequence_3": ["set_3", "set_2", "set_1", "set_4"],
    "sequence_4": ["set_4", "set_2", "set_3", "set_1"],
}
