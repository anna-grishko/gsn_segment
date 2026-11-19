 #let report(
  title: "title",
  datetime: "2026-11-03 17:23:34",
  body,
) = {

 set text(
    size: 14pt,
  )

 set page(
  "us-letter",
  margin: (left: 0.5in, right: 0.5in, top: 0.85in, bottom: 0.55in),
  background: place(top, rect(fill: aqua, width: 100%, height: 0.85in)),
  header: align(left, text(size: 20pt, fill: white, weight: "bold", title)),
  footer: context align(
      grid(
        columns: (80%, 20%),
        align(left, text(size: 16pt, datetime)),
        align(right, counter(page).display("1/1", both: true,)),
      ),
    ),
  )
  body
}