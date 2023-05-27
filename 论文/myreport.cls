\ProvidesClass{myreport}
\LoadClass{ctexart}
% \LoadClass{ctexrep}
\RequirePackage{xeCJK}
\RequirePackage{CJKnumb}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}
\RequirePackage{enumitem}
\RequirePackage{listings}
\RequirePackage{xcolor}
\RequirePackage{array}
\RequirePackage{mathtools}
\RequirePackage{float}
\RequirePackage{indentfirst}
\RequirePackage{zhnumber}
\RequirePackage{booktabs}
\RequirePackage{listings}
\RequirePackage{color}
\RequirePackage{xcolor}
\RequirePackage{pdfpages}
\RequirePackage{optidef}
\RequirePackage{subfigure}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{chngcntr}
\RequirePackage{multirow}

% 伪代码
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\renewcommand{\algorithmicrequire}{\textbf{Input:}}
\renewcommand{\algorithmicensure}{\textbf{Output:}}
% 跳转引用
\RequirePackage{xcolor}
\definecolor{winered}{rgb}{0.5,0,0}
\RequirePackage{hyperref}
\hypersetup{
  pdfborder={0 0 0},
  colorlinks=true,
  linkcolor={winered},
  urlcolor={winered},
  filecolor={winered},
  citecolor={winered},
  linktoc=all,
}


% 参考文献
\RequirePackage[
  backend=biber,
  citestyle=numeric,
  bibstyle=numeric,
  natbib=true,
  sorting=none]{biblatex}


\linespread{1.2}
\geometry{top=1in,bottom=1in,left=1in,right=1in}
\setcounter{secnumdepth}{3}
\renewcommand\thesection{\CJKnumber{\arabic{section}}、}
\renewcommand\thesubsection{\arabic{subsection}.}
\renewcommand\thesubsubsection{\arabic{subsection}.\arabic{subsubsection}}
\setlist[enumerate,1]{label=(\arabic*)}
\setlist[enumerate,2]{label=\alph*.}

\setCJKmainfont[ItalicFont={KaiTi},BoldFont={SimHei}]{SimSun}

\titleformat*{\section}{\large\bfseries}
\titleformat*{\subsection}{\normalsize\bfseries}
\titleformat*{\subsubsection}{\normalsize}
\graphicspath{{figures/}{figure/}{pictures/}%
{picture/}{pic/}{pics/}{image/}{images/}}

% \titlespacing{\subsection}{1em}{*4}{*1.5}
% \titlespacing{\subsubsection}{1em}{*4}{*1.5}

% 代码格式和颜色定义
\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{comment}{rgb}{0.56,0.64,0.68}
\lstset{
  frame=tb,
  aboveskip=3mm,
  belowskip=3mm,
  xleftmargin=2em,
  xrightmargin=2em,
  showstringspaces=false,
  columns=flexible,
  framerule=1pt,
  rulecolor=\color{gray!35},
  backgroundcolor=\color{gray!5},
  basicstyle={\small\ttfamily},
  numbers=none,
  numberstyle=\tiny\color{gray},
  keywordstyle=\color{blue},
  commentstyle=\color{comment},
  stringstyle=\color{dkgreen},
  breaklines=true,
  breakatwhitespace=true,
  tabsize=2,
}

% 信息
\newcommand\major[1]{\def\@major{#1}}
\newcommand\name[1]{\def\@name{#1}}
\newcommand\stuid[1]{\def\@stuid{#1}}
\newcommand\college[1]{\def\@college{#1}}
\renewcommand\date[1]{\def\@date{#1}}
\newcommand\course[1]{\def\@course{#1}}
\newcommand\subcoverpic[1]{\def\@subcoverpic{#1}}
\newcommand\coverpic[1]{\def\@coverpic{#1}}

% 页眉
\pagestyle{fancy}
\lhead{\@title}
\chead{姓名：\@name}
\rhead{学号：\@stuid}

% 下划线定义
\def\xjtuu[#1]#2{
  \underline{\hbox to #1{\hfil#2\hfil}}}


% 封面
\newcommand{\makecover}{
  \begin{center}
    \begin{center}
      \includegraphics[width=0.8\linewidth]{cover \@coverpic}
    \end{center}
    \vspace{1cm}
    \bfseries\huge{\@title}\\
    \vspace{0.5cm}
    \begin{center}
      \includegraphics[width=0.4\linewidth]{\@subcoverpic}
    \end{center}
    \vspace{1cm}
    \begin{center}
      \Large
      \linespread{2}
      \centering
      \renewcommand\arraystretch{1.6}
      \begin{tabular}{p{3cm}p{6cm}}
        \bfseries{课程名称：} & \@course \\ \cline{2-2}
        \bfseries{姓名：}     & \@name   \\ \cline{2-2}
        \bfseries{学院：}     & \@college \\ \cline{2-2}
        \bfseries{专业：}     & \@major  \\ \cline{2-2}
        \bfseries{学号：}     & \@stuid  \\ \cline{2-2}
      \end{tabular}
    \end{center}
    \vspace{0.5cm}
    \Large{\@date}
  \end{center}
  \thispagestyle{empty}
  \newpage
}

%Python code
\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{mauve}{rgb}{0.58,0,0.82}

\lstset{frame=tb,
  language=Python,
  aboveskip=3mm,
  belowskip=3mm,
  showstringspaces=false,
  columns=flexible,
  basicstyle={\small\ttfamily},
  numbers=none,
  numberstyle=\tiny\color{gray},
  keywordstyle=\color{blue},
  commentstyle=\color{dkgreen},
  stringstyle=\color{mauve},
  breaklines=true,
  breakatwhitespace=true,
  tabsize=3
}

%R code
\lstset{numbers=left,
        numberstyle=\tiny,
        keywordstyle=\color{orange},
        commentstyle=\color[cmyk]{1,0,1,0},
        stringstyle=\color{dkgreen},
        frame=single,
        escapeinside=``,
        extendedchars=false,
        xleftmargin=2em,xrightmargin=2em, aboveskip=1em,
        tabsize=4,
        showspaces=false
       }


% 允许伪代码分页
\makeatletter
\newenvironment{breakablealgorithm}
  {% \begin{breakablealgorithm}
   \begin{center}
     \refstepcounter{algorithm}% New algorithm
     \hrule height.8pt depth0pt \kern2pt% \@fs@pre for \@fs@ruled
     \renewcommand{\caption}[2][\relax]{% Make a new \caption
       {\raggedright\textbf{\ALG@name~\thealgorithm} ##2\par}%
       \ifx\relax##1\relax % #1 is \relax
         \addcontentsline{loa}{algorithm}{\protect\numberline{\thealgorithm}##2}%
       \else % #1 is not \relax
         \addcontentsline{loa}{algorithm}{\protect\numberline{\thealgorithm}##1}%
       \fi
       \kern2pt\hrule\kern2pt
     }
  }{% \end{breakablealgorithm}
     \kern2pt\hrule\relax% \@fs@post for \@fs@ruled
   \end{center}
  }
\makeatother