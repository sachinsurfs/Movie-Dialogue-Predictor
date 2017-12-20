for file in *.dot; do
	dot -Tpng "$file" -o "$(basename "$file" .dot).png" 
    #mv "$file" "$(basename "$file" .html).dot"
    done

