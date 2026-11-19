#show: report.with(
  $if(title)$
    title: "$title$",
  $endif$
  $if(datetime)$
    datetime: "$datetime$",
  $endif$
)