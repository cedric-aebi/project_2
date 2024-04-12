# How to use this template
This is a template for creating short reports based on the bfh-ci package using the bfhpub class. It helps to get started quickly. Problems and improvements can be reported via our issue tracker. Any kind of improvement is welcome, also other templates that cover a use case based on bfhpub class.

## Installing the dependencies
Install a recent LaTeX distribution and the latest version of the LaTeX package bfh-ci. How to do this is described in detail in the online manual. Follow the instructions for your operating system.
 * https://latex.ti.bfh.ch

## Information about the structure
The "*.tex" is a LaTeX document with a preamble and a LaTeX body with placeholder text. We recommend that you use this example as a quick introduction to writing short reports, such as a lab report. For an example of how to insert citations to online articles, books, etc., we recommend using biblatex. The bibliography database is stored in a ".bib" file. The content of such a file follows the definitions described in the biblatex user manual. Use the sample.bib file to get started.

## Parameterization
There is a local `latexmk` configuration file called ".latexmkrc". This file contains some pre-configurations like the name of the output directory ("_build") and the default latex compiler ("lualatex"). Modify or delete this file.

## Note
We recommend using Git for version control. If you are using Git, create a gitignore file with a good set of ignore patterns. For a quick start, we recommend the following patterns.
 * Get a general set of ignore patterns
   * https://www.toptal.com/developers/gitignore?templates=latex,windows,linux,vim,emacs,osx
  ```
  curl -L -o .gitignore https://www.toptal.com/developers/gitignore/api/latex,windows,linux,vim,emacs,osx
  ```
 * Ignore all directories that start with an underscore '_*/'
  ```
  echo '_*/' >> .gitignore
  ``` 

## Compiling
To compile the LaTeX document, use your favorite LaTeX editor together with the TeX compiler (xelatex and lualatex are supported). Thre is no support for pdflatex compiler.

If you use latexmk on the command line, the following command will do the compilation.

#### Running Latexmk
 * In the simplest case you just have to type
```bash
latexmk -lualatex
```
 *This runs LaTeX on all .tex files in the current directory using the output format specified by the configuration files.*

 * If you want to make sure to get a .pdf file as output, just mention it:
```bash
latexmk -pdflua
```
 * If you want to compile only one specific .tex file in the current directory, just provide the file name:
```bash
latexmk -lualatex myfile.tex
```
 * If you want to preview the resulting output file(s), just use
```bash
latexmk -pv -lualatex
```
 * And now the killer feature: If you want LaTeXmk to continuously check all input files for changes and re-compile the whole thing if needed and always display the result, type
```bash
latexmk -pvc -lualatex -interaction=nonstopmode
```
Then, whenever you change something in any of your source files and save your changes, the preview is automatically updated. But: This doesn’t work with all viewers, especially not with Adobe Reader. See the section about configuration files below for setting a suitable viewer application.

 * Of course, options can be combined, e.g.
```bash
latexmk -outdir=_build -pdf -pv myfile.tex
```

#### Cleanup
  * After running LaTeX, the current directory is contaminated with a myriad of temporary files; you can get rid of them with
```bash
latexmk -c
```
  * Previous command doesn’t delete the final .pdf/.ps/.dvi files. If you want to delete those too, use
```bash
latexmk -C
```
