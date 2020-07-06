
#include <vector>
#include "TaskFactory.h"
#include "SendTask.h"






int main(void) {
    std::vector<std::unique_ptr<Task>> apps;
    apps.push_back(TaskFactory::Create("SEND"));
    for (auto it = apps.begin(); it != apps.end(); ++it){
        it->get()->Execute();
    }
}

