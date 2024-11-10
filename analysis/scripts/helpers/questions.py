glob_big5_questions = [
             "I am someone who is outgoing, sociable",
             "I am someone who is compassionate, has a soft heart",
             "I am someone who tends to be disorganized",
             "I am someone who is relaxed, handles stress well",
             "I am someone who has few artistic interests",
             "I am someone who has an assertive personality",
             "I am someone who is respectful, treats others with respect",
             "I am someone who tends to be lazy",
             "I am someone who stays optimistic after experiencing a setback",
             "I am someone who is curious about many different things",
             "I am someone who rarely feels excited or eager",
             "I am someone who tends to find fault with others",
             "I am someone who is dependable, steady",
             "I am someone who is moody, has up and down mood swings",
             "I am someone who is inventive, finds clever ways to do things",
             "I am someone who tends to be quiet",
             "I am someone who feels little sympathy for others",
             "I am someone who is systematic, likes to keep things in order",
             "I am someone who can be tense",
             "I am someone who is fascinated by art, music, or literature",
             "I am someone who is dominant, acts as a leader",
             "I am someone who starts arguments with others",
             "I am someone who has difficulty getting started on tasks",
             "I am someone who feels secure, comfortable with self",
             "I am someone who avoids intellectual, philosophical discussions",
             "I am someone who is less active than other people",
             "I am someone who has a forgiving nature",
             "I am someone who can be somewhat careless",
             "I am someone who is emotionally stable, not easily upset",
             "I am someone who has little creativity",
             "I am someone who is sometimes shy, introverted",
             "I am someone who is helpful and unselfish with others",
             "I am someone who keeps things neat and tidy",
             "I am someone who worries a lot",
             "I am someone who values art and beauty",
             "I am someone who finds it hard to influence people",
             "I am someone who is sometimes rude to others",
             "I am someone who is efficient, gets things done",
             "I am someone who often feels sad",
             "I am someone who is complex, a deep thinker",
             "I am someone who is full of energy",
             "I am someone who is suspicious of others' intentions",
             "I am someone who is reliable, can always be counted on",
             "I am someone who keeps their emotions under control",
             "I am someone who has difficulty imagining things",
             "I am someone who is talkative",
             "I am someone who can be cold and uncaring",
             "I am someone who leaves a mess, doesn't clean up",
             "I am someone who rarely feels anxious or afraid",
             "I am someone who thinks poetry and plays are boring",
             "I am someone who prefers to have others take charge",
             "I am someone who is polite, courteous to others",
             "I am someone who is persistent, works until the task is finished",
             "I am someone who tends to feel depressed, blue",
             "I am someone who has little interest in abstract ideas",
             "I am someone who shows a lot of enthusiasm",
             "I am someone who assumes the best about people",
             "I am someone who sometimes behaves irresponsibly",
             "I am someone who is temperamental, gets emotional easily",
             "I am someone who is original, comes up with new ideas",
             ]

glob_reversed_questions = [11, 16, 26, 31, 36, 51, 12, 17, 22, 37, 42, 47, 3, 8, 23, 28, 48, 58, 4, 9, 24, 29, 44, 49, 5, 25, 30, 45, 50, 55]
glob_reversed_questions.sort()
glob_all_columns = ["bfi{}".format(str(i)) if i not in glob_reversed_questions else "rbfi{}".format(str(i)) for i in range(1, 61)]
glob_normal_columns =  ["bfi{}".format(str(i)) for i in range(1, 61)  if i not in glob_reversed_questions]
glob_reversed_columns =  ["rbfi{}".format(str(i)) for i in range(1, 61)  if i in glob_reversed_questions]

glob_normal_likert = {"Disagree strongly" : 1, "Disagree" : 2, "Neutral" : 3, "Agree" : 4, "Agree strongly" : 5}
glob_reverse_likert = {"Disagree strongly" : 5, "Disagree" : 4, "Neutral" : 3, "Agree" : 2, "Agree strongly" : 1}

glob_normal_likert_numbers = {"1" : "Disagree strongly", "2" : "Disagree", "3" : "Neutral", "4" : "Agree", "5" : "Agree strongly"}
glob_reverse_likert_numbers = {"5" : "Disagree strongly", "4" : "Disagree", "3" : "Neutral", "2" : "Agree", "1" : "Agree strongly"}

glob_gt_map = {"Lie": 1, "Half-truth": 0.5, "Truth": 0}