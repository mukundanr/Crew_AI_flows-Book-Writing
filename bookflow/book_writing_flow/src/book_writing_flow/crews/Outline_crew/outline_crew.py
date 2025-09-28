from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from pydantic import BaseModel

class Outline(BaseModel):
    """Outline of the book"""
    total_chapters: int
    titles: list[str]


@CrewBase
class OutlineCrew:
    """Outline Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def research_agent(self) -> Agent:
        return Agent(config=self.agents_config["research_agent"],
                     tools=[SerperDevTool()])

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])
    
    @agent
    def outline_writer(self) -> Agent:
        return Agent(config=self.agents_config["outline_writer"])

    @task
    def write_outline(self) -> Task:
        return Task(config=self.tasks_config["write_outline"],
                    output_pydantic=Outline)

    @crew
    def crew(self) -> Crew:

        return Crew(agents=self.agents,
                    tasks=self.tasks,
                    process=Process.sequential,
                    verbose=True)