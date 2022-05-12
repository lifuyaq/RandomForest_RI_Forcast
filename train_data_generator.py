import datetime

g_shift_hours = None
file = "interval.txt"


class Typhoon:
    def __init__(self):
        self.data = []
        self.sample_hours = 6.0

    def get_shift_data(self):
        shift_num = int(g_shift_hours / self.sample_hours)
        for i in range(len(self.data) - shift_num):
            # self.data[i].append(self.data[i + shift_num][4])
            self.data[i].append(self.data[i + shift_num][5])
            yield self.data[i]

    def feed_data(self, data_line):
        # if len(self.data) > 0 and self.sample_hours is None:
        #     latest = self.data[-1][0]
        #     self.sample_hours = (data_line[0] - latest).total_seconds() / 3600
        #     self.sample_hours = 6.0             # sample hour is 6.0
        self.data.append(data_line)

    def is_mine(self, data_line):
        if len(self.data) > 0:
            latest = self.data[-1][0]
            sample_hours = (data_line[0] - latest).total_seconds() / 3600
            if sample_hours != self.sample_hours:
                # print("WARNING: sampling time is incorrect, there might be any data missing!")
                # if self.data[-1][5] >= 15 or data_line[5] >= 15:
                #     print(self.data[-1][0], self.data[-1][5])
                #     print(data_line[0], data_line[5])
                if 12 <= sample_hours <= 24:
                    # print(self.data[-1][0], self.data[-1][5])
                    # print(data_line[0], data_line[5])
                    with open(file, "a") as f:
                        print(sample_hours, file=f)
                lat = data_line[2]
                lon = data_line[3]
                latest_lat = self.data[-1][2]
                latest_lon = self.data[-1][3]
                if abs(latest_lat - lat) > 4 or abs(latest_lon - lon) > 4:
                    return False
            return True
        else:
            return True


class Scanner:
    def __init__(self):
        self.current = None

    def feed_line(self, data_line):
        ret = None
        if self.current is None:
            self.current = Typhoon()
            self.current.feed_data(data_line)
        elif self.current.is_mine(data_line):
            self.current.feed_data(data_line)
        else:
            ret = self.current
            self.current = Typhoon()
            self.current.feed_data(data_line)
        return ret


def generate_train_data(container, output):
    with open(output, "w") as f:
        for typhoon in container:
            for data_line in typhoon.get_shift_data():
                line = data_line[0].strftime("%Y%m%d%H") + ' ' + ' '.join([str(d) for d in data_line[1:]]) + '\n'
                f.write(line)


def scan_data_source(input):
    container = []
    scanner = Scanner()
    with open(input, "r") as f:
        for line in f.readlines():
            fields = [float(d) for d in line.strip().split(' ')]
            data_line = [datetime.datetime.strptime(f"{int(fields[0])}", "%Y%m%d%H")]
            data_line += [int(d) for d in fields[1:6]]
            data_line += fields[6:]
            typhoon = scanner.feed_line(data_line)
            if typhoon is not None:
                container.append(typhoon)
        container.append(scanner.current)
    return container


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='./factor')
    parser.add_argument("--output", default='./132.txt')
    parser.add_argument("--shift_hours", default=132)
    args = parser.parse_args()
    g_shift_hours = args.shift_hours
    data = scan_data_source(args.input)
    generate_train_data(data, args.output)
