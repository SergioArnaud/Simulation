from numpy.random import exponential,uniform, normal, gamma
from heapq import heappush, heappop , heapify

def inventory_model(lambda_ = 10, unif_params = (4,9), demand_distribution = normal,
                    demand_params = (300,100) , months = 100, s = 100, S=1000, 
                    fixed_monthly_cost = 100, unit_cost= 3.5):
    
    def get_transactions(heap):
        
        k = 1
        order_arrival_time = exponential(lambda_)
        order_deliver_time = order_arrival_time + uniform(*unif_params)

        while order_deliver_time < days:
            demand = round(demand_distribution(*demand_params),2)
            heappush(heap, ( order_deliver_time, 'Delivery {}'.format(k), demand))

            k += 1
            order_arrival_time += exponential(lambda_)
            order_deliver_time = order_arrival_time + uniform(*unif_params)
            
        return heap
        
    monthly_cost = []
    
    days = 30*months
    last_time = 0
    
    inventory_level = S
    I1,I2 = 0,0
    
    heap = [(30*i,'Reorder inventory', 0) for i in range(1,months)]
    heapify(heap)
    heap = get_transactions(heap)
    
    while heap:
        time, operation, quantity = heappop(heap)
        
        if inventory_level > 0:
            I1 += (time-last_time)*(inventory_level)/30
        else:
            I2 += (time-last_time)*(-inventory_level)/30
        
        if 'Reorder' in operation:
            if inventory_level < s:
                monthly_cost.append(fixed_monthly_cost + unit_cost*(S-inventory_level))
                inventory_level = S
            else:
                monthly_cost.append(fixed_monthly_cost)
        else:
            inventory_level -= quantity
        
        last_time = time
        
            
    return I1/months, I2/months, sum(monthly_cost)

if __name__ == "__main__":

	inventory_model()