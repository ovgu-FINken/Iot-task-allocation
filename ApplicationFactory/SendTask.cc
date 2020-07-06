#include "SendTask.h"
#include <iostream>

void SendTask::Execute(){
    std::cout << "Running Sending Task";
}
std::unique_ptr<Task> SendTask::CreateMethod(){
    return std::make_unique<SendTask>();
}
std::string SendTask::GetFactoryName(){
    return "SEND";
}

bool SendTask::s_registered = TaskFactory::Register(SendTask::GetFactoryName(), SendTask::CreateMethod);
