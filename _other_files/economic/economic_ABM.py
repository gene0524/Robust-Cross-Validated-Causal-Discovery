import numpy as np
import random

class Firm:
    def __init__(self, id, type):
        self.id = id
        self.type = type  # 'consumption' or 'capital'
        self.capital = 100
        self.employees = []
        self.productivity = 1
        self.inventory = 0
        self.price = 1

    def innovate(self):
        if random.random() < 0.05:  # 5% 機率創新
            self.productivity *= 1.1

    def produce(self):
        production = self.productivity * len(self.employees)
        self.inventory += production
        return production

    def set_price(self, market_share):
        self.price = self.price * (1 + 0.1 * (market_share - 0.5))

class Worker:
    def __init__(self, id):
        self.id = id
        self.employed = False
        self.employer = None
        self.wage = 1
        self.savings = 10

    def consume(self, firms):
        consumption = min(self.wage * 0.9, self.savings)
        self.savings -= consumption
        # 隨機選擇一家消費品企業購買商品
        firm = random.choice([f for f in firms if f.type == 'consumption'])
        purchased = min(consumption / firm.price, firm.inventory)
        firm.inventory -= purchased

class Bank:
    def __init__(self):
        self.deposits = 0
        self.loans = 0

    def provide_loan(self, firm, amount):
        self.loans += amount
        firm.capital += amount

class Government:
    def __init__(self):
        self.budget = 1000

    def collect_taxes(self, firms, workers):
        for firm in firms:
            self.budget += 0.1 * firm.capital
        for worker in workers:
            self.budget += 0.1 * worker.wage

    def provide_benefits(self, unemployed_workers):
        benefit = 0.4  # 失業救濟金為工資的40%
        for worker in unemployed_workers:
            if self.budget > benefit:
                worker.savings += benefit
                self.budget -= benefit

class Economy:
    def __init__(self, num_firms, num_workers):
        self.firms = [Firm(i, 'consumption') for i in range(num_firms//2)] + \
                     [Firm(i+num_firms//2, 'capital') for i in range(num_firms//2)]
        self.workers = [Worker(i) for i in range(num_workers)]
        self.bank = Bank()
        self.government = Government()

    def run_simulation(self, num_periods):
        for period in range(num_periods):
            # 企業創新
            for firm in self.firms:
                firm.innovate()

            # 生產決策
            for firm in self.firms:
                firm.produce()

            # 勞動市場互動
            unemployed = [w for w in self.workers if not w.employed]
            for firm in self.firms:
                while len(firm.employees) < firm.capital // 10 and unemployed:
                    worker = random.choice(unemployed)
                    firm.employees.append(worker)
                    worker.employed = True
                    worker.employer = firm
                    unemployed.remove(worker)

            # 消費決策
            for worker in self.workers:
                worker.consume(self.firms)

            # 投資決策
            for firm in self.firms:
                if firm.capital < 100 and random.random() < 0.1:
                    self.bank.provide_loan(firm, 50)

            # 政府行為
            self.government.collect_taxes(self.firms, self.workers)
            self.government.provide_benefits([w for w in self.workers if not w.employed])

            # 更新市場份額和價格
            total_production = sum(f.produce() for f in self.firms)
            for firm in self.firms:
                market_share = firm.produce() / total_production
                firm.set_price(market_share)

            # 打印一些統計數據
            print(f"Period {period}:")
            print(f"Total production: {total_production}")
            print(f"Unemployment rate: {len([w for w in self.workers if not w.employed]) / len(self.workers)}")
            print(f"Average firm capital: {sum(f.capital for f in self.firms) / len(self.firms)}")
            print("---")

# 運行模擬
economy = Economy(num_firms=10, num_workers=100)
economy.run_simulation(num_periods=50)