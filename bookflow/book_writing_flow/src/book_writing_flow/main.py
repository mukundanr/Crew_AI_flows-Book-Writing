#!/usr/bin/env python
import os
import asyncio
from pathlib import Path

from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from book_writing_flow.crews.Writer_crew.writer_crew import ChapterWriterCrew
from book_writing_flow.crews.Outline_crew.outline_crew import OutlineCrew
from dotenv import load_dotenv

load_dotenv()

class Chapter(BaseModel):
    title: str = ""
    content: str = ""

class BookState(BaseModel):
    topic: str = "Retrieval Augmented Generation (RAG) - Overview and Advancements"
    total_chapters: int = 0
    titles: list[str] = []
    chapters: list[Chapter] = []

class ChapterFlow(Flow[BookState]):

    @start()
    def generate_outline(self):
        print("Generating outline")
        
        # generate outline on the "topic"
        outline = OutlineCrew().crew().kickoff(inputs={"topic": self.state.topic})
        
        # collect total chapters and chapter titles from the crew output
        self.state.total_chapters = outline.pydantic.total_chapters
        self.state.titles = outline.pydantic.titles

    @listen(generate_outline)
    async def generate_chapters(self):
        print("Generating chapters")
        tasks = []
    
        async def write_single_chapter(title: str):
            result = (
                ChapterWriterCrew()
                .crew()
                .kickoff(inputs={
                    "title": title,
                    "topic": self.state.topic,
                    "chapters": [chapter.title for chapter in self.state.chapters]
                })
            )
            return result.pydantic
    
        # Create tasks for each chapter
        for i in range(self.state.total_chapters):
            task = asyncio.create_task(write_single_chapter(self.state.titles[i]))
            tasks.append(task)
    
        # Wait for all chapters to be generated concurrently
        chapters = await asyncio.gather(*tasks)
        print(f"Generated {len(chapters)} chapters")
        self.state.chapters.extend(chapters)

    @listen(generate_chapters)
    def save_book(self):
        print("Saving book")
        base_dir = Path.cwd()
        path = base_dir / ("book.md")
        with path.open("w", encoding="utf-8") as f:
            for chapter in self.state.chapters:
                f.write("# " + chapter.title + "\n")
                f.write(chapter.content + "\n")

def kickoff():
    book_flow = ChapterFlow()
    asyncio.run(book_flow.kickoff_async())

def plot():
    book_flow = ChapterFlow()
    book_flow.plot()

if __name__ == "__main__":
    kickoff()